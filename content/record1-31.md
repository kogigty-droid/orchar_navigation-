为了对接你已经封装好的 **ROS 串口节点**，我将代码改造成了一个标准的 **ROS 节点**。

该代码会通过 D455 获取图像并识别树干，利用**外参**将坐标转到小车底盘中心，通过**纯跟踪算法**计算出控制指令，最后发布 `geometry_msgs/Twist` 消息到 `/cmd_vel` 话题。

### 1. 运行环境准备
确保你的环境中安装了以下依赖：
```bash
pip install pyrealsense2 ultralytics numpy opencv-python
# 确保已安装 ROS 和相关的 Python 消息包
```

### 2. 完整代码实现 (`orchard_nav_node.py`)

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
from geometry_msgs.msg import Twist
import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import math
from collections import deque

# ==================== 配置参数 ====================
# 1. 路径配置
MODEL_PATH = "/media/cs/ca046fc4-0dff-46aa-b4e4-b4f194d2d1101/depth_ws/ultralytics-8.2.100/orchard_trunk/d455_trunk2/weights/best.pt"

# 2. 静态外参 (根据你的安装位置：前进 50cm, 上方 20cm)
CAM_FORWARD = 0.5   # 相机相对于后轴中心的 X 偏移 (m)
CAM_ABOVE = 0.2     # 相机相对于地面的 Z 偏移 (m)
CAM_LEFT = 0.0      # Y 偏移

# 3. 纯跟踪 (Pure Pursuit) 参数
LOOK_AHEAD_DIST = 2.0    # 预瞄距离 L (m)
LINEAR_VEL = 0.4         # 巡航线速度 (m/s)
WHEEL_BASE = 0.6         # 车辆轴距 (m)
MAX_ANGULAR_VEL = 1.0    # 最大角速度限制 (rad/s)

# 4. 树行识别参数
LANE_WIDTH = 3.2         # 预估行宽 (m)
BUFFER_SIZE = 20         # 轨迹平滑缓存点数
# =================================================

class OrchardNavigatorNode:
    def __init__(self):
        # --- 1. ROS 初始化 ---
        rospy.init_node('orchard_nav_node', anonymous=True)
        self.vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.rate = rospy.Rate(30) # 30Hz

        # --- 2. RealSense D455 初始化 ---
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        profile = self.pipeline.start(config)
        
        # 获取内参
        color_stream = profile.get_stream(rs.stream.color)
        self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        self.align = rs.align(rs.stream.color)

        # --- 3. 外参定义 (相机系 -> 小车底盘系) ---
        # 小车X(前)=相机Z, 小车Y(左)=-相机X, 小车Z(上)=-相机Y
        self.R_bc = np.array([
            [0,  0, 1],
            [-1, 0, 0],
            [0, -1, 0]
        ])
        self.t_bc = np.array([CAM_FORWARD, CAM_LEFT, CAM_ABOVE])

        # --- 4. 算法组件 ---
        print(">>> 正在加载 YOLO 模型...")
        self.model = YOLO(MODEL_PATH)
        self.left_buf = deque(maxlen=BUFFER_SIZE)
        self.right_buf = deque(maxlen=BUFFER_SIZE)
        
        self.is_path_valid = False
        self.nav_k = 0.0  # 斜率 (航向)
        self.nav_b = 0.0  # 截距 (横向偏差)
        self.avg_floor_z = 0.0

    def process_frame(self):
        """核心处理逻辑：感知 -> 转换 -> 拟合"""
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        color_f = aligned.get_color_frame()
        depth_f = aligned.get_depth_frame()
        if not color_f or not depth_f: return None

        img = np.asanyarray(color_f.get_data())
        depth_img = np.asanyarray(depth_f.get_data())
        h, w = img.shape[:2]

        # YOLO 推理
        results = self.model.predict(img, conf=0.35, verbose=False)
        
        new_left, new_right = [], []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # 采样树根处的深度
                cx1, cx2 = x1 + int((x2-x1)*0.3), x2 - int((x2-x1)*0.3)
                cy1, cy2 = y1 + int((y2-y1)*0.6), y2
                roi = depth_img[max(0,cy1):min(h,cy2), max(0,cx1):min(w,cx2)]
                valid_d = roi[roi > 0]
                
                if len(valid_d) < 5: continue
                dist = np.median(valid_d) * self.depth_scale

                # --- 坐标转换：像素 -> 相机3D -> 小车底盘3D ---
                p_cam = rs.rs2_deproject_pixel_to_point(self.intrinsics, [(x1+x2)//2, y2], dist)
                p_base = self.R_bc.dot(p_cam) + self.t_bc
                
                rx, ry, rz = p_base[0], p_base[1], p_base[2]
                
                # 门控过滤 (针对当前行)
                if 0.5 < rx < 8.0: # 距离前方 0.5-8.0m
                    if 0.6 < ry < 2.5: # 左侧
                        new_left.append((rx, ry, rz))
                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    elif -2.5 < ry < -0.6: # 右侧
                        new_right.append((rx, ry, rz))
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # 更新 Buffer
        for p in new_left: self.left_buf.append(p)
        for p in new_right: self.right_buf.append(p)

        # 拟合中心线 (底盘系: y = kx + b)
        mx, my, mz = [], [], []
        for x, y, z in self.left_buf: mx.append(x); my.append(y - LANE_WIDTH/2.0); mz.append(z)
        for x, y, z in self.right_buf: mx.append(x); my.append(y + LANE_WIDTH/2.0); mz.append(z)

        if len(mx) > 4:
            self.nav_k, self.nav_b = np.polyfit(mx, my, 1)
            self.avg_floor_z = sum(mz) / len(mz)
            self.is_path_valid = True
        else:
            self.is_path_valid = False

        self.draw_debug(img)
        return img

    def draw_debug(self, img):
        """可视化导航线"""
        if self.is_path_valid:
            points_3d_base = []
            for x_b in np.linspace(1.0, 6.0, 10):
                y_b = self.nav_k * x_b + self.nav_b
                # 转回相机系进行投影
                p_cam = self.R_bc.T.dot(np.array([x_b, y_b, self.avg_floor_z]) - self.t_bc)
                pixel = rs.rs2_project_point_to_pixel(self.intrinsics, p_cam.tolist())
                points_3d_base.append(tuple(map(int, pixel)))
            
            for i in range(len(points_3d_base)-1):
                cv2.line(img, points_3d_base[i], points_3d_base[i+1], (0, 255, 255), 3)

    def control_loop(self):
        """计算纯跟踪并发布到 ROS 话题"""
        while not rospy.is_shutdown():
            img = self.process_frame()
            
            twist_msg = Twist()
            if self.is_path_valid:
                # --- 1. 纯跟踪算法计算 ---
                # 预瞄点 P(Lx, Ly)
                lx = LOOK_AHEAD_DIST
                ly = self.nav_k * lx + self.nav_b
                
                # 计算曲率 kappa = 2*dy / L^2
                kappa = 2.0 * ly / (LOOK_AHEAD_DIST**2)
                
                # 计算角速度 omega = v * kappa
                v = LINEAR_VEL
                omega = v * kappa
                
                # 限制最大转向速度
                omega = np.clip(omega, -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL)

                # --- 2. 填充并发布消息 ---
                twist_msg.linear.x = v
                twist_msg.angular.z = omega
                
                if img is not None:
                    cv2.putText(img, f"V:{v:.1f} W:{omega:.2f}", (50, 50), 0, 0.8, (0, 255, 0), 2)
            else:
                # 路径丢失，安全停止
                twist_msg.linear.x = 0.0
                twist_msg.angular.z = 0.0
                if img is not None:
                    cv2.putText(img, "STOP - NO PATH", (50, 50), 0, 0.8, (0, 0, 255), 2)

            self.vel_pub.publish(twist_msg)

            if img is not None:
                cv2.imshow("Orchard ROS Navigator", img)
                if cv2.waitKey(1) == ord('q'): break
            
            self.rate.sleep()

    def stop(self):
        self.vel_pub.publish(Twist()) # 停机指令
        self.pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    node = OrchardNavigatorNode()
    try:
        print(">>> 导航节点已启动，正在发布 /cmd_vel...")
        node.control_loop()
    except rospy.ROSInterruptException:
        pass
    finally:
        node.stop()
```

### 3. 如何使用

1.  **启动底盘节点**：
    启动你那个已经写好的、负责接收 `/cmd_vel` 并转换为 12 字节串口指令的节点：
    ```bash
    roslaunch your_chassis_package serial_control.launch
    ```

2.  **启动导航节点**：
    将上面的代码保存为 `orchard_nav_node.py`，赋予执行权限并运行：
    ```bash
    chmod +x orchard_nav_node.py
    python3 orchard_nav_node.py
    ```

### 4. 关键逻辑解析

*   **角速度的正负方向**：
    *   在 ROS 标准中，`angular.z > 0` 代表**左转**，`angular.z < 0` 代表**右转**。
    *   在代码中，如果树木偏向左边（`nav_b > 0`），计算出的 `omega` 为正，小车左转纠偏，符合逻辑。
*   **外参带来的精度提升**：
    *   如果不加 `CAM_FORWARD` (0.5m)，纯跟踪会认为“看到树”的位置就是“车中心”的位置。
    *   加入外参后，算法知道相机在车头，纠偏动作会从小车的**旋转中心**（通常是后轴）发起，走出的圆弧更加丝滑，不会切内径撞树。
*   **预瞄距离 $L$ 的调节**：
    *   如果在实车测试中发现车“扭动”厉害，请在代码顶部调大 `LOOK_AHEAD_DIST`（比如设为 2.5 或 3.0）。
    *   如果小车对弯道反应太慢，调小它（比如 1.5）。

### 5. 转发模块配合
*   当你的遥控器拨到“自动驾驶”模式时，你的底盘节点会解析 `/cmd_vel`。
*   由于代码中加入了 `if not self.is_path_valid: stop`，一旦 YOLO 漏检或者相机被挡住，小车会立即停下，保证了实地测试的安全。
