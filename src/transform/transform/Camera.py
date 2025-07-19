import cv2
import numpy as np
import os

class Camera:
    def __init__(self, camera_index=0, calib_file='camera_calib.npz'):
        self.cap = cv2.VideoCapture(camera_index)
        self.calib_file = calib_file
        self.camera_matrix = None
        self.dist_coeffs = None
    #    self.load_calibration()
        #self.check_camera()

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            raise IOError("无法接收图像帧")
        return frame
    
    def check_camera(self):
        if not self.cap.isOpened():
            raise IOError("无法打开摄像头")
        print("摄像头已成功打开")
    def release(self):
        self.cap.release()

    def __del__(self):
        self.release()

""" def load_calibration(self):
        # 加载相机标定参数 
        if not os.path.exists(self.calib_file):
            raise FileNotFoundError(f"标定文件 {self.calib_file} 不存在，请先运行标定程序生成该文件。")
        
        with np.load(self.calib_file) as calib_data:
            self.camera_matrix = calib_data['camera_matrix']
            self.dist_coeffs = calib_data['dist_coeffs']
        print("相机标定参数加载成功")"""


""" def get_camera_matrix(self):
        # 返回相机内参矩阵 
        if self.camera_matrix is None:
            raise ValueError("相机内参矩阵未加载")
        return self.camera_matrix"""

"""    def get_distortion_coefficients(self):
        # 返回畸变系数 
        if self.dist_coeffs is None:
            raise ValueError("畸变系数未加载")
        return self.dist_coeffs"""

