import cv2
import cv2.aruco as aruco
import numpy as np
import pyrealsense2 as rs
import yaml

# 从 YAML 文件中读取相机的内外参数
def load_camera_parameters(filename):
    with open(filename, 'r') as infile:
        data = yaml.safe_load(infile)
    camera_matrix = np.array(data['camera_matrix'], dtype=np.float32)
    dist_coeffs = np.array(data['dist_coeffs'], dtype=np.float32)
    return camera_matrix, dist_coeffs


# 计算物体距离并实时显示
def detect_aruco_and_show_distance(camera_matrix, dist_coeffs):
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    aruco_parameters = aruco.DetectorParameters_create()

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    profile = pipeline.start(config)
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = aruco.detectMarkers(gray_image, aruco_dict, parameters=aruco_parameters)

            if ids is not None:
                aruco.drawDetectedMarkers(color_image, corners, ids)

                rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(corners, 0.1, camera_matrix, dist_coeffs)

                for i in range(len(ids)):
                    aruco.drawAxis(color_image, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.1)

                    # Get depth value
                    x, y = int(corners[i][0][0][0]), int(corners[i][0][0][1])
                    depth = depth_image[y, x] * depth_scale * 100
                    cv2.putText(color_image, f"Distance: {depth:.2f}cm", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 0, 0), 2)
            cv2.imshow('RealSense L515', color_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    camera_matrix, dist_coeffs = load_camera_parameters('L515.yaml')
    detect_aruco_and_show_distance(camera_matrix, dist_coeffs)
