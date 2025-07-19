import cv2
import numpy as np

class Transform:
    def __init__(self):
        pass

    def detect(self, frame):
        try:
            # 处理图像，获取处理后的帧、黄色掩码和角点列表
            processed_frame, yellow_mask, corner_points_list = self.process_frame(frame)

            # 默认 warped_img 为输入图像的副本，防止未赋值
            warped_img = frame.copy()

            # 如果检测到角点，进行透视变换
            if corner_points_list:
                for idx, corner_points in enumerate(corner_points_list):
                    # 透视变换目标尺寸
                    target_width, target_height = 200, 200

                    # 定义目标图像的四个角点
                    target_points = np.array([
                        [0, 0],
                        [target_width, 0],
                        [target_width, target_height],
                        [0, target_height]
                    ], dtype=np.float32)

                    # 确保是 4 个点
                    if len(corner_points) == 4:
                        src_points = np.array(corner_points, dtype=np.float32)

                        # 计算透视变换矩阵
                        H, mask = cv2.findHomography(src_points, target_points, cv2.RANSAC, 5.0)

                        if H is not None:
                            # 应用透视变换
                            warped_img = cv2.warpPerspective(frame, H, (target_width, target_height))
                            break  # 只处理第一个有效的四边形
                        else:
                            print("Homography 矩阵计算失败，使用默认图像")
                    else:
                        print(f"角点数量错误: {len(corner_points)}，需为 4")
            else:
                print("未检测到角点，使用默认图像")

            return processed_frame, yellow_mask, warped_img

        except Exception as e:
            print(f"Transform.detect 错误: {e}")
            # 返回默认图像以防止节点崩溃
            return frame.copy(), np.zeros_like(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)), frame.copy()

    def preprocess_image(self, img, resize_shape=(640, 480), brightness_factor=5, threshold_val=50):
        img_resized = cv2.resize(img, resize_shape)
        img_dark = cv2.convertScaleAbs(img_resized, alpha=brightness_factor)
        img_gray = cv2.cvtColor(img_dark, cv2.COLOR_BGR2GRAY)
        _, img_binary = cv2.threshold(img_gray, threshold_val, 255, cv2.THRESH_BINARY)
        return img_dark, img_binary

    def process_frame(self, frame):
        resized_img, binary_img = self.preprocess_image(frame)

        hsv = cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array([23, 40, 40])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # 形态学操作：去噪 & 合并小块
        yellow_mask = cv2.dilate(yellow_mask, None, iterations=5)

        contours, _ = cv2.findContours(yellow_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        corner_points_list = []  # 存储所有四边形角点列表

        # 添加相机内参和畸变参数
        camera_matrix = np.array([
            [600, 0, 320],
            [0, 600, 240],
            [0, 0, 1]
        ], dtype=np.float32)
        dist_coeffs = np.zeros((5, 1))  # 忽略畸变

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 5000:
                # 获取最小外接矩形
                rect = cv2.minAreaRect(cnt)
                box = cv2.boxPoints(rect)  # 获取矩形的4个顶点坐标
                box = np.int32(box)  # 转换为整数坐标

                # 绘制最小外接矩形
                cv2.drawContours(resized_img, [box], 0, (255, 0, 0), 2)

                # 存储当前矩形的角点
                corner_points = []
                for i, point in enumerate(box):
                    x, y = point
                    corner_points.append((x, y))
                    # 绘制红点
                    cv2.circle(resized_img, (x, y), radius=5, color=(0, 0, 255), thickness=-1)
                    # 编号文本
                    cv2.putText(resized_img, f"{i}", (x + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # 添加到总列表中
                corner_points_list.append(corner_points)

                # 输出角点坐标
                if corner_points_list:
                    print("检测到的角点坐标（像素）：")
                    for idx, points in enumerate(corner_points_list):
                        print(f"  四边形 {idx}: {points}")

        return resized_img, yellow_mask, corner_points_list