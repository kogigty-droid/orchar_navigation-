完整代码实现 (Orchard 3D Tracking)
3D 门控：只保留物理坐标在当前行范围内的树（解决多行干扰）。
点云 Buffer：使用 deque 存储最近 30 个有效 3D 点（解决抖动）。
AR 可视化：把拟合出的数学直线，投影回屏幕画出来（解决可视化）。
控制接口：计算出k,b供纯跟踪使用。

# 代码： orchard_realtime_nav.py

这是一个完全针对**实车部署**、**去除录像包逻辑**、并严格落实你要求的**3D门控+Buffer防抖+AR可视化**的完整代码。

这段代码是直接连接 RealSense 相机进行实时处理的，**没有 Bag 文件回放逻辑**。

### 代码设计核心逻辑：
1.  **3D 门控 (Physical Gating)**：在获得 XYZ 坐标后，立即判断 $X$ 是否在“行宽”范围内。如果 $X$ 太大（比如 > 2.5m），说明是隔壁行的树，直接丢弃，**绝不进入 Buffer**。
2.  **点云 Buffer (Memory)**：使用 `collections.deque` 存储最近 30 个**有效的** 3D 坐标点。即使当前帧 YOLO 漏检，Buffer 里还有之前的 29 个点支撑拟合，路径不会消失或跳变。
3.  **AR 可视化 (Projection)**：利用 `rs.rs2_project_point_to_pixel`，将拟合出的数学直线 $X=kZ+b$ 反向投影回屏幕，画出贴地的导航线。

### 实车运行代码 (`orchard_realtime_nav.py`)

请确保 USB 线已连接 D455，且模型路径正确。

```python
import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import math
from collections import deque

# ==================== 实车配置参数 ====================
# 1. 模型路径 (请确认为你实车工控机上的真实路径)
SEG_MODEL_PATH = "/media/cs/ca046fc4-0dff-46aa-b4e4-b4f194d2d1101/depth_ws/road_best.pt" 
DET_MODEL_PATH = "/media/cs/ca046fc4-0dff-46aa-b4e4-b4f194d2d1101/depth_ws/ultralytics-8.2.100/orchard_trunk/d455_trunk2/weights/best.pt"

# 2. 3D 物理门控 (单位: 米)
# 假设行宽约 3.2米，车在中间，树干距离相机左右约 1.6米。
# 我们设置一个物理围栏：
# 左边树有效范围: X 在 [-2.5, -0.5] 之间
# 右边树有效范围: X 在 [0.5, 2.5] 之间
# 超过这个范围的(比如 X=4.0)，绝对是隔壁行，直接剔除！
GATE_X_MIN_LEFT = -2.8
GATE_X_MAX_LEFT = -0.5
GATE_X_MIN_RIGHT = 0.5
GATE_X_MAX_RIGHT = 2.8

# Z轴过滤 (太近深度不准，太远看不清)
ROI_Z_MIN = 0.4
ROI_Z_MAX = 8.0

# 3. 稳定性参数
BUFFER_SIZE = 30          # 记忆最近 30 个有效点 (约1秒数据)
LANE_WIDTH_GUESS = 3.2    # 用于将左右点归一化到中心线
CONF_THRES = 0.35         # 稍微提高阈值，实车环境宁缺毋滥

# ====================================================

class OrchardRealTimeNavigator:
    def __init__(self):
        print(">>> 正在初始化 RealSense D455 相机...")
        # 1. 初始化相机 (实车模式)
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # D455 推荐配置：高帧率、低延迟
        self.config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
        
        # 启动流
        self.profile = self.pipeline.start(self.config)
        
        # 获取内参 (用于 3D<->2D 转换)
        color_stream = self.profile.get_stream(rs.stream.color)
        self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        
        # 深度对齐 (必须对齐，否则像素对应不上)
        self.align = rs.align(rs.stream.color)

        print(">>> 正在加载 YOLO 模型...")
        self.model = YOLO(DET_MODEL_PATH)

        # 2. 初始化 Buffer (双端队列)
        # 里面存的是元组: (z, x, y_floor)
        self.left_buffer = deque(maxlen=BUFFER_SIZE)
        self.right_buffer = deque(maxlen=BUFFER_SIZE)

        # 导航参数状态
        self.nav_k = 0.0
        self.nav_b = 0.0
        self.is_path_valid = False

    def process(self):
        try:
            # 1. 获取实时帧
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)
            
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            
            if not color_frame or not depth_frame:
                return None

            img_color = np.asanyarray(color_frame.get_data())
            img_depth = np.asanyarray(depth_frame.get_data())
            h, w = img_color.shape[:2]

            # 2. YOLO 推理
            results = self.model.predict(img_color, conf=CONF_THRES, verbose=False)

            # 临时列表 (本帧的新点)
            new_left_points = []
            new_right_points = []

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    # === 深度提取策略：取树根 ===
                    # 取检测框下半部分的中心区域，避开边缘背景
                    crop_w = x2 - x1
                    crop_h = y2 - y1
                    cx1 = max(0, x1 + int(crop_w * 0.25))
                    cx2 = min(w, x2 - int(crop_w * 0.25))
                    cy1 = max(0, y1 + int(crop_h * 0.5)) # 只看下半截
                    cy2 = min(h, y2)

                    depth_roi = img_depth[cy1:cy2, cx1:cx2]
                    valid_d = depth_roi[depth_roi > 0]

                    # 如果有效深度点太少，说明这里可能是黑洞或者没对准
                    if len(valid_d) < 10: 
                        continue

                    # 取中位数距离
                    dist = np.median(valid_d) * self.depth_scale

                    # === 坐标转换 (2D -> 3D) ===
                    # 取框底部中心作为接地点
                    cx, cy = (x1 + x2) // 2, y2
                    # rs2_deproject: 像素(u,v) + 深度 -> 物理坐标(x,y,z)
                    # point_3d = [x(右), y(下), z(前)]
                    point_3d = rs.rs2_deproject_pixel_to_point(self.intrinsics, [cx, cy], dist)
                    
                    real_x = point_3d[0]
                    real_y = point_3d[1] # 这是高度 (相对于相机)，用于后续画线更贴地
                    real_z = point_3d[2]

                    # === 【关键步骤】3D 门控 (Physical Gating) ===
                    # 这里是解决多行干扰、跳变的核心！
                    # 只有在物理围栏内的点，才有资格进入 Buffer
                    
                    if ROI_Z_MIN < real_z < ROI_Z_MAX:
                        if GATE_X_MIN_LEFT < real_x < GATE_X_MAX_LEFT:
                            # 判定为当前行-左侧树
                            new_left_points.append((real_z, real_x, real_y))
                            # 绘制绿框
                            cv2.rectangle(img_color, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(img_color, f"{real_x:.2f}m", (x1, y1-5), 0, 0.5, (0, 255, 0), 1)
                            
                        elif GATE_X_MIN_RIGHT < real_x < GATE_X_MAX_RIGHT:
                            # 判定为当前行-右侧树
                            new_right_points.append((real_z, real_x, real_y))
                            # 绘制蓝框
                            cv2.rectangle(img_color, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            cv2.putText(img_color, f"{real_x:.2f}m", (x1, y1-5), 0, 0.5, (255, 0, 0), 1)
                            
                        else:
                            # 判定为隔壁行/干扰项 -> 丢弃！
                            # 绘制灰框，表示“我看到你了，但我不理你”
                            cv2.rectangle(img_color, (x1, y1), (x2, y2), (100, 100, 100), 1)
                            # cv2.putText(img_color, f"Ignored {real_x:.1f}", (x1, y1-5), 0, 0.5, (100, 100, 100), 1)

            # 3. 更新 Buffer (FIFO 队列)
            # 只有通过门控的点才会进来，保证 Buffer 里全是干净的数据
            for pt in new_left_points: self.left_buffer.append(pt)
            for pt in new_right_points: self.right_buffer.append(pt)

            # 4. 路径拟合 & 控制参数计算
            self.calculate_path()

            # 5. AR 可视化 (投影回屏幕)
            self.draw_ar_overlay(img_color)

            return img_color

        except Exception as e:
            print(f"Frame Process Error: {e}")
            return None

    def calculate_path(self):
        """
        利用 Buffer 中的所有点，拟合中心线
        """
        # 将左、右树干归一化到"路中心"
        mid_z = []
        mid_x = []
        floor_y_list = [] # 用于计算地面高度，方便画图

        # 左树 -> 往右平移半个行宽
        for z, x, y in self.left_buffer:
            mid_z.append(z)
            mid_x.append(x + LANE_WIDTH_GUESS / 2.0)
            floor_y_list.append(y)

        # 右树 -> 往左平移半个行宽
        for z, x, y in self.right_buffer:
            mid_z.append(z)
            mid_x.append(x - LANE_WIDTH_GUESS / 2.0)
            floor_y_list.append(y)

        # 至少要有足够的点才能拟合
        # 注意：因为有 Buffer，所以即使当前帧没检测到，这里依然可能有几十个点
        if len(mid_z) >= 4:
            # 最小二乘法拟合直线 X = kZ + b
            self.nav_k, self.nav_b = np.polyfit(mid_z, mid_x, 1)
            self.is_path_valid = True
            
            # 计算平均地面高度 (用于 AR 画线)
            self.avg_floor_y = sum(floor_y_list) / len(floor_y_list)
        else:
            # Buffer 空了，说明很久没看到树了
            self.is_path_valid = False

    def draw_ar_overlay(self, img):
        """
        AR 可视化：将拟合出的 3D 线画回 2D 屏幕
        """
        if not self.is_path_valid:
            cv2.putText(img, "NO PATH - SEARCHING", (50, 50), 0, 1, (0, 0, 255), 2)
            return

        # 1. 生成 Z 轴上的采样点 (从车头前 1米 到 10米)
        z_points = np.linspace(1.0, 10.0, 10)
        pixel_points = []

        for z in z_points:
            # 根据方程算出 X
            x = self.nav_k * z + self.nav_b
            
            # 构造 3D 点: [x, y_floor, z]
            # y_floor 使用统计出来的平均树根高度，这样线就像贴在地上一样
            point_3d = [x, self.avg_floor_y, z]
            
            # === 核心：3D -> 2D 投影 ===
            pixel = rs.rs2_project_point_to_pixel(self.intrinsics, point_3d)
            pixel_points.append(tuple(map(int, pixel)))

        # 2. 连线 (画黄色粗线作为路径)
        for i in range(len(pixel_points) - 1):
            p1 = pixel_points[i]
            p2 = pixel_points[i+1]
            
            # 简单的屏幕边界检查
            if 0<=p1[0]<img.shape[1] and 0<=p1[1]<img.shape[0]:
                cv2.line(img, p1, p2, (0, 255, 255), 4)

        # 3. 显示控制参数 (调试用)
        # 这两个值 (k, b) 就是你纯跟踪需要的输入
        info = f"Offset(b): {self.nav_b:.2f}m | Angle(k): {math.degrees(math.atan(self.nav_k)):.1f}deg"
        cv2.putText(img, info, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    def close(self):
        self.pipeline.stop()

# ==================== 主程序入口 ====================
if __name__ == "__main__":
    nav = OrchardRealTimeNavigator()
    print(">>> 系统启动！按 Q 退出")
    
    try:
        while True:
            # 处理一帧
            frame = nav.process()
            
            if frame is not None:
                cv2.imshow("Orchard Navigation Real-Time", frame)
            
            # 这里可以添加把 nav.nav_k 和 nav.nav_b 发送给纯跟踪控制器的代码
            # controller.compute(nav.nav_k, nav.nav_b) ...

            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        nav.close()
        cv2.destroyAllWindows()
```

### 代码使用说明与注意事项：

1.  **实车参数调整（非常重要！）**：
    *   **`GATE_X`**：这是解决“多行干扰”的命门。拿着卷尺量一下你的行宽。
        *   如果是 3.2米行宽：车在中间，树离车心 1.6米。
        *   `GATE` 设为 `[0.5, 2.5]` 是合理的（允许 1.6米左右波动）。
        *   **如果隔壁行的树（比如 4.8米处）被画了绿框，说明你的 MAX 值太大了，必须改小！**
    *   **`LANE_WIDTH_GUESS`**：这个值用于把左右两边的树归一化到中间。填真实行宽即可。

2.  **Buffer 的作用**：
    *   代码里 `self.left_buffer` 是一个 `deque`。
    *   如果某一帧车颠簸了一下，YOLO 没检测到树，或者检测到的树因为超出门控被剔除了。
    *   **没关系！** Buffer 里还存着上一帧、上上帧...的 30 个点。
    *   `fit_path` 函数会用这 30 个老点依然拟合出一条非常稳定的线。这就是**“防抖”**的原理。

3.  **AR 可视化验证**：
    *   上车后，先把车摆正。看屏幕上的**黄线**。
    *   黄线应该稳稳地贴在两行树的正中间，延伸向远方。
    *   让人去推一下车（模拟抖动），黄线应该**不会**跟着画面剧烈跳动，而是有一种“粘滞感”（因为有 Buffer 平滑）。

4.  **对接纯跟踪**：
    *   代码里计算出的 `self.nav_k` 和 `self.nav_b` 就是你需要的控制输入。
    *   $k$ 代表车身偏角（斜率）。
    *   $b$ 代表车身横向偏差（截距）。
    *   直接把这两个数喂给你之前的纯跟踪函数即可算出角速度。
