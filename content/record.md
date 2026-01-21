# orchard navigation

### **“防呆”和“防抖”**处理
### 主要修改点：
#### 1.深度提取：改为取框的下半部分（树根）。
#### 2.近距离补救：深度丢失时，根据像素坐标估算距离。
#### 3.角速度平滑：增加了 output_filter，防止舵机打摆子。

<details>
<summary>orchard_track_robust_v2.py</summary>
    
```python
import sys
# 适配 ROS 环境
sys.path.append('/opt/ros/noetic/lib/python3/dist-packages')

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Int8
import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import math
from collections import deque

# ==================== 实车调优参数 ====================
# 1. 硬件
BAG_FILE = None
SEG_MODEL_PATH = "/media/cs/ca046fc4-0dff-46aa-b4e4-b4f194d2d1101/depth_ws/road_best.pt"
DET_MODEL_PATH = "/media/cs/ca046fc4-0dff-46aa-b4e4-b4f194d2d1101/depth_ws/ultralytics-8.2.100/orchard_trunk/d455_trunk2/weights/best.pt"

# 2. 视觉过滤 (放宽了限制，防止丢树)
ROI_Z_MIN, ROI_Z_MAX = 0.25, 10.0 # 允许更近的物体(配合几何估算)
ROI_X_MAX_WIDTH = 4.5             # 允许更宽的视野
CONF_THRES = 0.25                 # 适当提高，减少误检
LANE_WIDTH_GUESS = 3.2            # 行距
WINDOW_SIZE = 15                  # 增大平滑窗口，抵抗转向抖动

# 3. 运动控制
TARGET_SPEED = 0.3        # 实车建议先慢一点
LOOKAHEAD_DIST = 2.5      # 【关键】加大预瞄距离！这能极大缓解转向画龙的问题
MAX_ANGULAR_VEL = 0.6     # 限制最大转向速度
SMOOTH_ALPHA = 0.2        # 输出滤波因子 (越小越平滑)
# ====================================================

class SmoothingPathPlanner:
    def __init__(self, window_size=10):
        self.history_k = deque(maxlen=window_size)
        self.history_b = deque(maxlen=window_size)
        self.stable_k = 0.0
        self.stable_b = 0.0
        self.is_initialized = False
        self.lost_frames = 0
        self.max_lost = 40 # 允许盲走 2秒 (20Hz)

    def update(self, left_pts, right_pts):
        mid_z, mid_x = [], []
        
        # 简单的数据关联
        for pt in left_pts:
            mid_z.append(pt[0]); mid_x.append(pt[1] + LANE_WIDTH_GUESS / 2.0)
        for pt in right_pts:
            mid_z.append(pt[0]); mid_x.append(pt[1] - LANE_WIDTH_GUESS / 2.0)

        current_k, current_b = None, None
        
        # 至少2个点拟合
        if len(mid_z) >= 2:
            current_k, current_b = np.polyfit(mid_z, mid_x, 1)
            self.lost_frames = 0
        
        # 只有1个点，采用惯性更新
        elif len(mid_z) == 1:
            if self.is_initialized:
                current_k = self.stable_k # 锁死斜率，防止旋转
                current_b = mid_x[0] - (current_k * mid_z[0])
                self.lost_frames = 0
        
        # 数据更新与平滑
        if current_k is not None:
            # 异常值剔除：如果这一帧突变太厉害，忽略
            if self.is_initialized and abs(current_b - self.stable_b) > 0.8:
                pass # 认为是噪声
            else:
                self.history_k.append(current_k)
                self.history_b.append(current_b)
                self.is_initialized = True
        else:
            self.lost_frames += 1

        if self.is_initialized and self.lost_frames < self.max_lost:
            self.stable_k = sum(self.history_k) / len(self.history_k)
            self.stable_b = sum(self.history_b) / len(self.history_b)
            return self.stable_k, self.stable_b, True
        else:
            return 0.0, 0.0, False

class PurePursuitController:
    def __init__(self):
        self.last_omega = 0.0

    def compute_cmd(self, k, b, speed_mps):
        # 1. 计算预瞄点
        goal_z = LOOKAHEAD_DIST
        goal_x = k * goal_z + b
        
        # 2. 纯跟踪曲率计算
        l_squared = goal_x**2 + goal_z**2
        curvature = 2.0 * goal_x / l_squared
        
        # 3. 计算目标角速度
        target_omega = speed_mps * curvature
        
        # 4. 【核心】输出平滑滤波 (Low Pass Filter)
        # 防止转向机打摆子，这一步对实车非常重要
        filtered_omega = self.last_omega * (1 - SMOOTH_ALPHA) + target_omega * SMOOTH_ALPHA
        
        # 5. 限幅
        filtered_omega = np.clip(filtered_omega, -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL)
        
        # 死区 (直线行驶时不动方向盘)
        if abs(filtered_omega) < 0.05: filtered_omega = 0.0
        
        self.last_omega = filtered_omega
        return (goal_x, goal_z), filtered_omega

def draw_3d_line(img, k, b, y_floor, intrinsics, color, thick=3):
    z_pts = np.linspace(1.0, 8.0, 5)
    px_pts = []
    for z in z_pts:
        x = k * z + b
        px = rs.rs2_project_point_to_pixel(intrinsics, [x, y_floor, z])
        px_pts.append(tuple(map(int, px)))
    for i in range(len(px_pts) - 1):
        p1, p2 = px_pts[i], px_pts[i+1]
        if 0<=p1[0]<img.shape[1] and 0<=p1[1]<img.shape[0]:
            cv2.line(img, p1, p2, color, thick)

def main():
    rospy.init_node('orchard_tracker_v2', anonymous=True)
    cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    work_pub = rospy.Publisher('/cmd_work', Int8, queue_size=10)
    rate = rospy.Rate(20)

    # 激活底盘
    rospy.sleep(1.0)
    work_msg = Int8(); work_msg.data = 1; work_pub.publish(work_msg)
    print("[ROS] Chassis Activated")

    print("Loading Models...")
    model_seg = YOLO(SEG_MODEL_PATH)
    model_det = YOLO(DET_MODEL_PATH)

    pipeline = rs.pipeline()
    config = rs.config()
    if BAG_FILE: rs.config.enable_device_from_file(config, BAG_FILE, repeat_playback=True)
    else:
        config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    align = rs.align(rs.stream.color)
    profile = pipeline.start(config)
    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    planner = SmoothingPathPlanner(window_size=WINDOW_SIZE)
    controller = PurePursuitController()

    print("READY. Press [SPACE] to Start/Stop, [Q] to Quit.")
    is_running = False 

    try:
        while not rospy.is_shutdown():
            try: frames = pipeline.wait_for_frames()
            except: break
            
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame: continue

            img_color = np.asanyarray(color_frame.get_data())
            img_depth = np.asanyarray(depth_frame.get_data())
            h, w, _ = img_color.shape

            # A. 检测处理
            det_results = model_det.predict(img_color, conf=CONF_THRES, verbose=False)
            valid_pts_left, valid_pts_right = [], []

            for r in det_results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # ==================== 【关键改进：深度提取 + 几何保底】 ====================
                    # 1. 区域选择：取框的【下半部分】(树根)，通常纹理更好，且对应真实位置
                    # 去掉顶部 50%，取下半部分的中间 50%
                    crop_h_start = y1 + int((y2-y1) * 0.5) 
                    crop_w_margin = int((x2-x1) * 0.25)
                    
                    cx1 = max(0, x1 + crop_w_margin)
                    cx2 = min(w, x2 - crop_w_margin)
                    cy1 = max(0, crop_h_start)
                    cy2 = min(h, y2)
                    
                    depth_roi = img_depth[cy1:cy2, cx1:cx2]
                    valid_d = depth_roi[depth_roi > 0]

                    dist = 0.0
                    use_geometric_backup = False

                    if len(valid_d) > 20:
                        dist = np.median(valid_d) * depth_scale
                        # 如果测出来的距离太近(<0.3)或太远(>15)，可能是噪声
                        if dist < 0.25 or dist > 15.0:
                            use_geometric_backup = True
                    else:
                        use_geometric_backup = True
                    
                    # 2. 【保底逻辑】：如果深度失效，用像素 Y 坐标估算距离
                    # 原理：树根越靠下 (y2越大)，离得越近。
                    # 这是一个粗略估算，但能防止近处树木丢失导致转向错误
                    if use_geometric_backup:
                        # 简单的地面投影模型 (假设相机高 0.8m)
                        # 这只是一个救命的 fallback，不需要极其精确
                        # 假设画面底部(y=720)是 0.3m，画面中心(y=360)是无穷远
                        normalized_y = (y2 - h/2) / (h/2) # 0~1
                        if normalized_y > 0.1:
                            dist = 0.8 / math.tan(normalized_y * 0.5) # 粗略公式
                            # 限制在一个合理范围内
                            dist = max(0.3, min(dist, 2.0))
                            
                            # 画个紫框表示这是“估算”出来的
                            cv2.rectangle(img_color, (x1, y1), (x2, y2), (255, 0, 255), 2)
                        else:
                            continue # 太高了，不可能是近处树根，丢弃

                    # 反投影
                    cx, cy = (x1+x2)//2, (y1+y2)//2
                    X, Y, Z = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], dist)
                    
                    # 空间过滤
                    if ROI_Z_MIN < Z < ROI_Z_MAX and abs(X) < ROI_X_MAX_WIDTH:
                        pt = (Z, X)
                        if X < 0: valid_pts_left.append(pt)
                        else: valid_pts_right.append(pt)
            
            # =========================================================================

            # 排序
            valid_pts_left.sort(key=lambda x: x[0])
            valid_pts_right.sort(key=lambda x: x[0])
            use_left = valid_pts_left[:TREE_COUNT_TO_FIT]
            use_right = valid_pts_right[:TREE_COUNT_TO_FIT]

            # B. 路径更新
            nav_k, nav_b, path_valid = planner.update(use_left, use_right)

            # C. 控制逻辑
            twist = Twist()
            
            if is_running and path_valid:
                # 计算纯跟踪
                (gx, gz), angular_z = controller.compute_cmd(nav_k, nav_b, TARGET_SPEED)
                
                twist.linear.x = TARGET_SPEED
                twist.angular.z = angular_z
                
                # 可视化
                draw_3d_line(img_color, nav_k, nav_b, 0.5, intrinsics, (0, 255, 255), 4)
                px = rs.rs2_project_point_to_pixel(intrinsics, [gx, 0.5, gz])
                if 0<=px[0]<w: cv2.circle(img_color, (int(px[0]), int(px[1])), 10, (0,0,255), -1)
                
                status_text = f"RUN: v={TARGET_SPEED} w={angular_z:.2f}"
                status_color = (0, 255, 0)
                
            elif is_running and not path_valid:
                twist.linear.x = 0.0; twist.angular.z = 0.0
                status_text = "LOST -> STOP"
                status_color = (0, 0, 255)
            else:
                twist.linear.x = 0.0; twist.angular.z = 0.0
                status_text = "PAUSED"
                status_color = (0, 255, 255)

            cmd_pub.publish(twist)

            # 显示检测到的点 (辅助调试)
            for p in use_left: 
                px = rs.rs2_project_point_to_pixel(intrinsics, [p[1], 0.5, p[0]])
                cv2.circle(img_color, (int(px[0]), int(px[1])), 5, (255,0,0), -1)
            for p in use_right:
                px = rs.rs2_project_point_to_pixel(intrinsics, [p[1], 0.5, p[0]])
                cv2.circle(img_color, (int(px[0]), int(px[1])), 5, (0,0,255), -1)

            cv2.putText(img_color, status_text, (20, 50), 0, 1, status_color, 2)
            cv2.imshow("Orchard Robust Track", img_color)

            key = cv2.waitKey(1)
            if key == ord('q'): break
            elif key == ord(' '): is_running = not is_running

            rate.sleep()

    except rospy.ROSInterruptException: pass
    finally:
        cmd_pub.publish(Twist()) # 停车
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```
</details>


#### 1.几何保底逻辑 (use_geometric_backup)：
问题：近处的树（右下角）因为太近，深度是黑的。
解决：代码检测到深度无效时，不直接丢弃，而是看它的框是否触底（y2 很大）。如果触底，强制给它赋一个近距离值（如 0.5m）。
效果：右下角那棵红树会被救回来，紫色的框会标记出来。这样在右转时，右边依然有“锚点”，路径不会飘。
#### 2.加大预瞄距离 (LOOKAHEAD_DIST = 2.5)：
问题：转向时，近处的点变化剧烈，如果盯着近处看，车会左右摇摆。
解决：把目光放远到 2.5 米外。那里的树在画面中相对稳定，拟合出的直线抖动小。纯跟踪算法会自动平滑掉近处的误差。
#### 3.强力输出滤波 (SMOOTH_ALPHA = 0.2)：
问题：路径计算难免会有噪声。
解决：不管感知识别怎么跳，最终发给底盘的命令被加上了“阻尼器”。底盘的转向动作会变得迟缓而平滑，不会因为一帧识别错误就猛打方向。




### 保持当前行运动的解决思路：
#### 1.几何边界锁定 (Geometric Bounding) —— 最简单、最快
#### 原理：利用果园的物理特性。行宽是固定的（比如 3.2米）。机器人只要在当前行里，计算出的横向偏差 (Offset b) 绝对不可能超过行宽的一半。
#### 逻辑：
1.你拟合出的路径方程是 X=kZ+b
2.b代表机器人距离路径中心线的横向距离。
3.如果计算出的 ∣b∣>1.6m (行宽3.2m的一半)，说明这条线肯定拟合到隔壁行去了，或者机器人已经偏离太远了。
4.措施：直接丢弃这一帧的计算结果，沿用上一帧的路径，或者触发“急停/重规划”。

#### 2.航向角速率限制 (Heading Rate Limit) —— 防止急转
#### 原理：在行间行驶时，机器人主要是走直线或微调。路径的角度（k值对应的角度）变化应该是连续且缓慢的。如果某一帧路径角度突然变了 20度，那肯定是因为连到了隔壁行的树。
#### 逻辑：
1.记录上一帧的路径角度 last_angle。
2.计算当前帧拟合出的角度 current_angle。
3.如果 abs(current_angle - last_angle) > 5度，判定为异常跳变。

#### 2.IMU 辅助锁定 (The "God" Compass) —— 最有效
#### 原理：视觉容易受骗（看着像路），但 IMU（陀螺仪） 不会撒谎。D455 相机内置了 IMU。如果视觉算法告诉你“前面右转 30度进入缺口”，但 IMU 告诉你“车身并没有转弯，且当前行进方向没变”，那么视觉大概率是错的。
#### 实现方法：
#### 既然你不想引入复杂的融合算法，我们可以用一个简单的逻辑：“基于 IMU 的航向保持”。
1.在进入行的一瞬间（或者按开始键时），记录当前的 IMU 航向角（Yaw）作为 基准航向 (Target Yaw)。
2.在行驶过程中，虽然允许视觉微调左右，但限制视觉规划出的路径不能偏离基准航向太多（比如 ±15度）。
3.如果视觉算出的路径要求转 30 度去追隔壁的树，直接被 IMU 逻辑否决。

### 解决“跑到隔壁行”最好的办法不是让模型识别得更准，而是给路径规划加“护栏”：
侧向护栏：算出来的路，偏离中心不能超过 1.5米。
角度护栏：算出来的路，转弯不能超过 20度。
盲走回正：一旦看不到树，不要保持转弯，而是慢慢把方向盘回正，尝试走直线找树。

<details>
<summary>orchard_nav_ultimate.py</summary>  
    
```python
import sys
# 适配 ROS 环境 (Ubuntu 20.04/Noetic)
sys.path.append('/opt/ros/noetic/lib/python3/dist-packages')

import rospy
from geometry_msgs.msg import Twist
from std_msgs.msg import Int8
import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import math
from collections import deque

# ==================== 终极调优参数 ====================
# 1. 硬件与模型
BAG_FILE = None # 实车设为 None
SEG_MODEL_PATH = "/media/cs/ca046fc4-0dff-46aa-b4e4-b4f194d2d1101/depth_ws/road_best.pt"
DET_MODEL_PATH = "/media/cs/ca046fc4-0dff-46aa-b4e4-b4f194d2d1101/depth_ws/ultralytics-8.2.100/orchard_trunk/d455_trunk2/weights/best.pt"

# 2. 视觉过滤 (感知层)
ROI_Z_MIN, ROI_Z_MAX = 0.25, 12.0 # 允许更近的树(配合几何估算)
ROI_X_MAX_WIDTH = 4.5             # 宽视野
CONF_THRES = 0.25                 # 平衡漏检与误检
LANE_WIDTH_GUESS = 3.2            # 真实行距
TREE_COUNT_TO_FIT = 6             # 拟合点数

# 3. 护栏策略 (控制层安全锁)
MAX_HEADING_ERROR = 25.0          # 【锁1】最大允许航向偏差(度)，超过认为异常
MAX_LATERAL_OFFSET = 1.5          # 【锁2】最大允许横向偏差(米)，超过认为跳行
WINDOW_SIZE = 15                  # 滑动窗口大小 (平滑感知跳变)

# 4. 运动控制 (执行层)
TARGET_SPEED = 0.3        # 巡航速度 (m/s)
LOOKAHEAD_DIST = 2.5      # 预瞄距离 (m)
MAX_ANGULAR_VEL = 0.6     # 最大角速度限制
SMOOTH_ALPHA = 0.15       # 输出滤波因子 (0.1~0.3，越小越平滑)
# ====================================================

class RobustPathPlanner:
    """
    带三重锁定策略的鲁棒路径规划器
    """
    def __init__(self, window_size=10):
        # 历史数据队列
        self.history_k = deque(maxlen=window_size)
        self.history_b = deque(maxlen=window_size)
        
        # 稳定输出值
        self.stable_k = 0.0
        self.stable_b = 0.0
        
        # 状态标志
        self.is_initialized = False
        self.lost_frames = 0
        self.max_lost = 40 # 允许盲走约2秒

    def update(self, left_pts, right_pts):
        """
        输入: 左右树干点集
        输出: k, b, is_valid
        """
        mid_z = []
        mid_x = []
        
        # 1. 数据关联：生成中点云
        for pt in left_pts:
            mid_z.append(pt[0]); mid_x.append(pt[1] + LANE_WIDTH_GUESS / 2.0)
        for pt in right_pts:
            mid_z.append(pt[0]); mid_x.append(pt[1] - LANE_WIDTH_GUESS / 2.0)

        # 2. 最小二乘拟合
        current_k, current_b = None, None
        
        # 至少2点拟合
        if len(mid_z) >= 2:
            current_k, current_b = np.polyfit(mid_z, mid_x, 1)
        
        # 单点惯性更新
        elif len(mid_z) == 1:
            if self.is_initialized:
                current_k = self.stable_k # 锁死斜率
                current_b = mid_x[0] - (current_k * mid_z[0])
        
        # 3. 【三重锁定策略】安全检查 (Gatekeeper)
        is_measurement_valid = False
        
        if current_k is not None:
            # 计算角度 (度)
            angle_deg = math.degrees(math.atan(current_k))
            
            # 【锁1】航向角限制：如果行间导航要求转弯 > 25度，肯定是连错树了
            if abs(angle_deg) > MAX_HEADING_ERROR:
                # print(f"拒绝: 角度过大 {angle_deg:.1f}")
                pass 
            
            # 【锁2】横向偏差限制：如果偏离中心 > 1.5米，肯定是跳行了
            elif abs(current_b) > MAX_LATERAL_OFFSET:
                # print(f"拒绝: 偏移过大 {current_b:.2f}")
                pass
            
            else:
                is_measurement_valid = True

        # 4. 状态更新
        if is_measurement_valid:
            self.lost_frames = 0
            
            # 异常值平滑 (简单的阶跃抑制)
            if self.is_initialized and abs(current_b - self.stable_b) > 0.8:
                # 如果突变太大，只更新一点点
                self.history_b.append(self.stable_b * 0.8 + current_b * 0.2)
                self.history_k.append(self.stable_k * 0.9 + current_k * 0.1)
            else:
                self.history_k.append(current_k)
                self.history_b.append(current_b)
            
            self.is_initialized = True
        else:
            self.lost_frames += 1

        # 5. 输出计算
        if self.is_initialized and self.lost_frames < self.max_lost:
            # 滑动窗口平均
            self.stable_k = sum(self.history_k) / len(self.history_k)
            self.stable_b = sum(self.history_b) / len(self.history_b)
            
            # 【锁3】盲走回正：如果丢失视野，强制让路径慢慢变直
            if self.lost_frames > 0:
                self.stable_k *= 0.95 
                
            return self.stable_k, self.stable_b, True
        else:
            return 0.0, 0.0, False

class PurePursuitController:
    def __init__(self):
        self.last_omega = 0.0

    def compute_cmd(self, k, b, speed_mps):
        # 1. 预瞄
        goal_z = LOOKAHEAD_DIST
        goal_x = k * goal_z + b
        
        # 2. 曲率
        l_squared = goal_x**2 + goal_z**2
        curvature = 2.0 * goal_x / l_squared
        
        # 3. 目标角速度
        target_omega = speed_mps * curvature
        
        # 4. 输出滤波 (Low Pass)
        filtered_omega = self.last_omega * (1 - SMOOTH_ALPHA) + target_omega * SMOOTH_ALPHA
        
        # 5. 限幅与死区
        filtered_omega = np.clip(filtered_omega, -MAX_ANGULAR_VEL, MAX_ANGULAR_VEL)
        if abs(filtered_omega) < 0.05: filtered_omega = 0.0
        
        self.last_omega = filtered_omega
        return (goal_x, goal_z), filtered_omega

def draw_3d_line(img, k, b, y_floor, intrinsics, color, thick=3):
    z_pts = np.linspace(1.0, 8.0, 5)
    px_pts = []
    for z in z_pts:
        x = k * z + b
        px = rs.rs2_project_point_to_pixel(intrinsics, [x, y_floor, z])
        px_pts.append(tuple(map(int, px)))
    for i in range(len(px_pts) - 1):
        p1, p2 = px_pts[i], px_pts[i+1]
        if 0<=p1[0]<img.shape[1] and 0<=p1[1]<img.shape[0]:
            cv2.line(img, p1, p2, color, thick)

def main():
    rospy.init_node('orchard_ultimate_driver', anonymous=True)
    cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
    work_pub = rospy.Publisher('/cmd_work', Int8, queue_size=10)
    rate = rospy.Rate(20)

    # 激活底盘
    rospy.sleep(1.0)
    work_msg = Int8(); work_msg.data = 1; work_pub.publish(work_msg)
    print("[ROS] Chassis Activated")

    print("Loading Models...")
    model_seg = YOLO(SEG_MODEL_PATH)
    model_det = YOLO(DET_MODEL_PATH)

    pipeline = rs.pipeline()
    config = rs.config()
    if BAG_FILE: rs.config.enable_device_from_file(config, BAG_FILE, repeat_playback=True)
    else:
        config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    align = rs.align(rs.stream.color)
    profile = pipeline.start(config)
    intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    # 初始化加强版规划器
    planner = RobustPathPlanner(window_size=WINDOW_SIZE)
    controller = PurePursuitController()

    print("ULTIMATE MODE READY. [SPACE] Start/Stop, [Q] Quit.")
    is_running = False 

    try:
        while not rospy.is_shutdown():
            try: frames = pipeline.wait_for_frames()
            except: break
            
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame: continue

            img_color = np.asanyarray(color_frame.get_data())
            img_depth = np.asanyarray(depth_frame.get_data())
            h, w, _ = img_color.shape

            # A. 感知层 (增强版深度提取)
            det_results = model_det.predict(img_color, conf=CONF_THRES, verbose=False)
            valid_pts_left, valid_pts_right = [], []

            for r in det_results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # 1. 深度提取策略: 取框的下半部分(树根)
                    crop_h_start = y1 + int((y2-y1) * 0.5) 
                    crop_w_margin = int((x2-x1) * 0.25)
                    cx1 = max(0, x1 + crop_w_margin)
                    cx2 = min(w, x2 - crop_w_margin)
                    cy1 = max(0, crop_h_start)
                    cy2 = min(h, y2)
                    
                    depth_roi = img_depth[cy1:cy2, cx1:cx2]
                    valid_d = depth_roi[depth_roi > 0]

                    dist = 0.0
                    use_geometric = False

                    if len(valid_d) > 20:
                        dist = np.median(valid_d) * depth_scale
                        if dist < 0.25 or dist > 15.0: use_geometric = True
                    else:
                        use_geometric = True
                    
                    # 2. 几何保底 (防止近处树木丢失)
                    if use_geometric:
                        normalized_y = (y2 - h/2) / (h/2) 
                        if normalized_y > 0.1:
                            dist = 0.8 / math.tan(normalized_y * 0.5) 
                            dist = max(0.3, min(dist, 2.0))
                            cv2.rectangle(img_color, (x1, y1), (x2, y2), (255, 0, 255), 2) # 紫框标记
                        else:
                            continue

                    # 反投影
                    cx, cy = (x1+x2)//2, (y1+y2)//2
                    X, Y, Z = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], dist)
                    
                    # 空间过滤
                    if ROI_Z_MIN < Z < ROI_Z_MAX and abs(X) < ROI_X_MAX_WIDTH:
                        pt = (Z, X)
                        if X < 0: valid_pts_left.append(pt)
                        else: valid_pts_right.append(pt)

            valid_pts_left.sort(key=lambda x: x[0])
            valid_pts_right.sort(key=lambda x: x[0])
            use_left = valid_pts_left[:TREE_COUNT_TO_FIT]
            use_right = valid_pts_right[:TREE_COUNT_TO_FIT]

            # B. 规划层 (Robust Update)
            nav_k, nav_b, path_valid = planner.update(use_left, use_right)

            # C. 控制层
            twist = Twist()
            
            if is_running and path_valid:
                (gx, gz), angular_z = controller.compute_cmd(nav_k, nav_b, TARGET_SPEED)
                
                twist.linear.x = TARGET_SPEED
                twist.angular.z = angular_z
                
                # 可视化
                draw_3d_line(img_color, nav_k, nav_b, 0.5, intrinsics, (0, 255, 255), 4)
                px = rs.rs2_project_point_to_pixel(intrinsics, [gx, 0.5, gz])
                if 0<=px[0]<w: cv2.circle(img_color, (int(px[0]), int(px[1])), 10, (0,0,255), -1)
                
                status_text = f"AUTO: {math.degrees(angular_z):.1f} rad/s"
                status_color = (0, 255, 0)
                
            elif is_running and not path_valid:
                twist.linear.x = 0.0; twist.angular.z = 0.0
                status_text = "LOST -> STOP"
                status_color = (0, 0, 255)
            else:
                twist.linear.x = 0.0; twist.angular.z = 0.0
                status_text = "PAUSED"
                status_color = (0, 255, 255)

            cmd_pub.publish(twist)

            # D. 显示
            for p in use_left: 
                px = rs.rs2_project_point_to_pixel(intrinsics, [p[1], 0.5, p[0]])
                cv2.circle(img_color, (int(px[0]), int(px[1])), 5, (255,0,0), -1)
            for p in use_right:
                px = rs.rs2_project_point_to_pixel(intrinsics, [p[1], 0.5, p[0]])
                cv2.circle(img_color, (int(px[0]), int(px[1])), 5, (0,0,255), -1)

            cv2.putText(img_color, status_text, (20, 50), 0, 1, status_color, 2)
            cv2.imshow("Orchard Ultimate", img_color)

            key = cv2.waitKey(1)
            if key == ord('q'): break
            elif key == ord(' '): is_running = not is_running

            rate.sleep()

    except rospy.ROSInterruptException: pass
    finally:
        cmd_pub.publish(Twist()) 
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```
</details>



# 2026-1-21

<img width="1219" height="450" alt="d3ec84491d2083fca9f5fe17c7ef1fcf" src="https://github.com/user-attachments/assets/01979797-ae14-4093-9d1f-4ed70368b836" />
