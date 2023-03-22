import cv2
import cv2.aruco as aruco
import numpy as np
import pyrealsense2 as rs
import yaml

# 生成Aruco标记并保存为图像文件
def generate_aruco_marker():
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    marker_size = 200
    marker = np.zeros((marker_size, marker_size), dtype=np.uint8)
    marker = aruco.drawMarker(aruco_dict, 1, marker_size, marker)
    cv2.imwrite('aruco_marker.png', marker)

# 使用Intel RealSense L515相机捕获图像
def capture_image():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    print("按 's' 拍照")

    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        cv2.imshow('RealSense L515', color_image)
        key = cv2.waitKey(1)

        if key == ord('s'):
            cv2.imwrite('captured_image.jpg', color_image)
            break

    pipeline.stop()
    cv2.destroyAllWindows()

# 检测Aruco标记并计算相机内外参数
def calculate_camera_parameters():
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    aruco_parameters = aruco.DetectorParameters_create()

    captured_image = cv2.imread('captured_image.jpg')
    gray_image = cv2.cvtColor(captured_image, cv2.COLOR_BGR2GRAY)

    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray_image, aruco_dict, parameters=aruco_parameters)

    if ids is not None:
        marker_length = 0.1  # 标记长度（以米为单位）
        camera_matrix = np.array([[640, 0, 320], [0, 480, 240], [0, 0, 1]])  # 估计相机内参数
        dist_coeffs = np.zeros((4, 1))  # 估计相机畸变系数

        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)

        return camera_matrix, dist_coeffs, rvecs, tvecs
    else:
        return None

# 将内外参数保存到YAML文件
def save_to_yaml(camera_matrix, dist_coeffs, rvecs, tvecs):
    data = {
        'camera_matrix': camera_matrix.tolist(),
        'dist_coeffs': dist_coeffs.tolist(),
        'rvecs': rvecs.tolist(),
        'tvecs': tvecs.tolist(),
    }

    with open('L515.yaml', 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

if __name__ == '__main__':
    generate_aruco_marker()
    capture_image()
    camera_parameters = calculate_camera_parameters()
if camera_parameters is not None:
    camera_matrix, dist_coeffs, rvecs, tvecs = camera_parameters
    save_to_yaml(camera_matrix, dist_coeffs, rvecs, tvecs)
    print("摄像头内外参数已保存到 L515.yaml")
else:
    print("未检测到Aruco标记，请重新拍照")
