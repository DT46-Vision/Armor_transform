#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from transform.transform import Transform

class TransformNode(Node):
    def __init__(self):
        super().__init__('transform_node')
        self.bridge = CvBridge()

        # 订阅/image_raw话题
        self.image_sub = self.create_subscription(
            Image, '/image_raw', self.image_callback, 10)

        # 创建发布者
        self.processed_pub = self.create_publisher(Image, 'processed_frame', 10)
        self.yellow_mask_pub = self.create_publisher(Image, 'yellow_mask', 10)
        self.warped_pub = self.create_publisher(Image, 'warped_img', 10)

        # 初始化Transform类
        self.trans = Transform()

        self.get_logger().info('Transform Node 已初始化')

    def image_callback(self, msg):
        try:
            # 将ROS图像消息转换为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # 使用Transform类的detect方法处理图像
            processed_frame, yellow_mask, warped_img = self.trans.detect(cv_image)

            # 将处理后的图像转换回ROS消息
            processed_msg = self.bridge.cv2_to_imgmsg(processed_frame, encoding='bgr8')
            yellow_mask_msg = self.bridge.cv2_to_imgmsg(yellow_mask, encoding='mono8')
            warped_msg = self.bridge.cv2_to_imgmsg(warped_img, encoding='bgr8')

            # 发布图像
            self.processed_pub.publish(processed_msg)
            self.yellow_mask_pub.publish(yellow_mask_msg)
            self.warped_pub.publish(warped_msg)

        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge 错误: {e}')
        except Exception as e:
            self.get_logger().error(f'图像处理错误: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = TransformNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Transform Node 已终止')
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()