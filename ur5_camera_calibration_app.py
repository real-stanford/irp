# %%
import argparse
import pathlib
import pickle

import numpy as np
import cv2
import pyzed.sl as sl
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

# %%
class Robot:
    def __init__(self, robot_ip, tcp_offset=0.5):
        tcp_offset_pose = [0.0]*6
        tcp_offset_pose[2] = tcp_offset
        self.tcp_offset_pose = tcp_offset_pose

        self.robot_ip = robot_ip
        self.rtde_c = None
        self.rtde_r = None
    
    def __enter__(self):
        self.rtde_c = RTDEControlInterface(self.robot_ip)
        self.rtde_r = RTDEReceiveInterface(self.robot_ip)
        self.rtde_c.setTcp(self.tcp_offset_pose)
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.rtde_c.stopScript()
        self.rtde_c.disconnect()
        self.rtde_r.disconnect()
    
    def get_tcp_point(self):
        tcp_pose = self.rtde_r.getActualTCPPose()
        tcp_point = tcp_pose[:3]
        return tcp_point

def deg_to_rad(deg):
    return deg / 180 * np.pi


def robot_init(robot):
    j_init = deg_to_rad(np.array([-90,-70,150,-170,-90,-90]))
    return robot.rtde_c.moveJ(
        j_init.tolist(), 1.0, 1.0, True)


def robot_move_joint(robot, joint_id, delta, acc=1.0, speed=1.0):
    new_j = np.array(robot.rtde_r.getActualQ())
    new_j[joint_id] += delta

    return robot.rtde_c.moveJ(
        new_j.tolist(), acc, speed, True)

class Camera:
    def __init__(self):
        self.camera = None
    
    def __enter__(self):
        init_params = sl.InitParameters(
            camera_resolution=sl.RESOLUTION.HD720,
            camera_fps=60,
            depth_mode=sl.DEPTH_MODE.NONE,
            sdk_verbose=True,
            sdk_gpu_id=0,
            enable_image_enhancement=True
        )
        zed = sl.Camera()
        err = zed.open(init_params)
        assert(err == sl.ERROR_CODE.SUCCESS)

        video_setting_dict = {
            'BRIGHTNESS': 4,
            'CONTRAST': 4,
            'HUE': 0,
            'SATURATION': 4,
            'SHARPNESS': 4,
            'GAMMA': 5,
            'AEC_AGC': 1,
            'WHITEBALANCE_AUTO': 1,
            'LED_STATUS': 1
        }

        for key, value in video_setting_dict.items():
            zed.set_camera_settings(getattr(sl.VIDEO_SETTINGS, key), value)

        self.camera = zed
        return self
    
    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.camera.close()
    
    def get_image(self):
        mat = sl.Mat()
        self.camera.grab()
        self.camera.retrieve_image(mat, sl.VIEW.LEFT)
        img = mat.get_data()
        return img


def homo_transform(mat, points):
    if mat.shape[1] != points.shape[1]:
        points = np.concatenate([points, np.ones((points.shape[0],1))], axis=1)
    homo = points @ mat.T
    result = (homo[:,:homo.shape[1]-1].T / homo[:,-1]).T
    return result


class UR5CalibrationApp:
    def __init__(self, camera, robot, window_name='left_image'):
        self.window_name = window_name
        self.current_img_point = None
        self.current_robot_point = None
        self.data = list()

        self.camera = camera
        self.robot = robot

    def start(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
    
    def end(self):
        cv2.destroyWindow(self.window_name)

    def mouse_callback(self, action, x, y, flags, *userdata):
        if action == cv2.EVENT_LBUTTONDOWN:
            coord = (x,y)
            print('click', coord)
            self.current_img_point = coord
    
    def step(self):
        img = self.camera.get_image()
        self.current_img = img.copy()
        self.draw_calibration(img)
        if self.current_img_point is not None:
            cv2.drawMarker(img, self.current_img_point, 
                color=(0,0,255), markerType=cv2.MARKER_CROSS,
                markerSize=20, thickness=1)
        cv2.imshow(self.window_name, img)
        key = cv2.waitKey(17)
        return key
    
    def update_robot_point(self):
        self.current_robot_point = self.robot.get_tcp_point()

    def clear_coord(self):
        self.current_img_point = None
        self.current_robot_point = None
    
    def save_coord(self):
        if self.current_robot_point is None:
            print("no robot point")
        elif self.current_img_point is None:
            print("no clicked point")
        else:
            self.data.append({
                'robot_point': self.current_robot_point,
                'img_point': self.current_img_point
            })
    
    def compute_homography(self):
        # assuming x won't change
        robot_points_2d = self.get_robot_points()[:,1:]
        img_points = self.get_image_points()
        # least squares
        tx_img_robot, confidence = cv2.findHomography(
            robot_points_2d, img_points)
        return tx_img_robot
    
    def get_robot_points(self):
        return np.array([x['robot_point'] for x in self.data])
    
    def get_image_points(self):
        return np.array([
            x['img_point'] for x in self.data], 
            dtype=np.float64)
    
    def __len__(self):
        return len(self.data)
    
    def draw_calibration(self, img):
        if len(self) >= 4:
            tx_img_robot = self.compute_homography()
            # draw box
            robot_corners = np.array([
                [1,0],
                [1,2],
                [-1,2],
                [-1,0]
            ], dtype=np.float64)
            img_conrers = homo_transform(
                tx_img_robot, robot_corners)
            cv2.polylines(img, 
                [img_conrers.round().astype(np.int32)], 
                isClosed=True, color=(255,0,0))
            # draw workspace
            robot_corners = np.array([
                [-3,-3],
                [3,-3],
                [3,3],
                [-3,3]
            ], dtype=np.float64)
            img_conrers = homo_transform(
                tx_img_robot, robot_corners)
            cv2.polylines(img, 
                [img_conrers.round().astype(np.int32)], 
                isClosed=True, color=(0,0,0))

            # draw robot points
            robot_points_2d = self.get_robot_points()[:,1:]
            val_img_points = homo_transform(
                tx_img_robot, robot_points_2d)
            for point in val_img_points:
                cv2.drawMarker(img, point.round().astype(np.int32), 
                color=(255,0,0), markerType=cv2.MARKER_CROSS,
                markerSize=20, thickness=1)            
        if len(self) > 0:
            # draw image points
            img_points = self.get_image_points()
            for point in img_points:
                cv2.drawMarker(img, point.round().astype(np.int32), 
                color=(0,255,0), markerType=cv2.MARKER_CROSS,
                markerSize=20, thickness=1)
    
    def save_calibration(self, fname):
        if len(self) < 4:
            print('Calibration requires at least 4 points!')
            return

        path = pathlib.Path(fname)
        path = path.with_suffix('.pkl')
        path.parent.mkdir(parents=True, exist_ok=True)
        calib_data = dict(self.data)
        calib_data['tx_img_robot'] = self.compute_homography()
        pickle.dump(calib_data, path.open('wb'))
        print(f"Calibration file saved to {path.absolute()}")
    
    def pop(self):
        if len(self.data) > 0:
            return self.data.pop()


# %%
def main():
    parser = argparse.ArgumentParser(
        'Planar calibration\nWASD control\nI to init robot\nQ to quit\n')
    parser.add_argument('-o', '--output', help='calibration file', required=True)
    parser.add_argument('--ip', default='192.168.0.139', help='robot ip')
    parser.add_argument('--delta', default=40, help="action delta in degrees")

    args = parser.parse_args()


    with Robot(robot_ip=args.ip) as robot:
        with Camera() as camera:
            app = UR5CalibrationApp(camera, robot)
            app.start()
            while True:
                delta = args.delta / 180 * 3.14
                key = app.step()
                if key == ord('q'):
                    print('exit')
                    app.save_calibration(args.output)
                    break
                elif key == ord('i'):
                    app.clear_coord()
                    robot_init(robot)
                elif key == ord('e'):
                    print('delete')
                    app.clear_coord()
                elif key == 13:
                    print('enter')
                    app.save_calibration(args.output)
                elif key == 8:
                    print('backspace')
                    app.pop()
                # move
                elif key == ord('w'):
                    app.save_coord()
                    app.clear_coord()
                    robot_move_joint(robot, 2, -delta)
                elif key == ord('s'):
                    app.save_coord()
                    app.clear_coord()
                    robot_move_joint(robot, 2, delta)
                elif key == ord('a'):
                    app.save_coord()
                    app.clear_coord()
                    robot_move_joint(robot, 3, -delta)
                elif key == ord('d'):
                    app.save_coord()
                    app.clear_coord()
                    robot_move_joint(robot, 3, delta)
                # if key != -1:
                #     print("key:", key)
                app.update_robot_point()
            
            app.end()
    return

# %%
if __name__ == '__main__':
    main()