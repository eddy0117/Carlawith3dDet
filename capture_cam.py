import carla
import numpy as np
import cv2
import threading
import queue
import time
from tools.thread_funcs import display_images, show_location, process_image, packed_data_send
from tools.utils import euler_to_rotation_matrix, cam_bp_maker

exit_event = threading.Event()


class CameraManager:
    def __init__(self, world, vehicle, cam_params):
        self.camera_arr = []
        self.img_queue_arr = []
        # self.camera_bp = camera_bp
        self.world = world
        self.tar_car = vehicle
        self.bp_lib = world.get_blueprint_library()
        self.cam_params = cam_params
        self.cam_intrinsics = None
        

    
    def set_cams(self):
        for cam_param in self.cam_params:
            cam_bp = cam_bp_maker(self.bp_lib, *cam_param[2])
            x, y, z = cam_param[0]
            roll, pitch, yaw = cam_param[1]
            cam_transform = carla.Transform(carla.Location(x=x, y=y, z=z), 
                                            carla.Rotation(roll=roll, pitch=pitch, yaw=yaw))
            self.camera_arr.append(self.world.spawn_actor(cam_bp, cam_transform, attach_to=self.tar_car))
            self.img_queue_arr.append(queue.Queue(maxsize=10))
        return self.camera_arr
    
    def listen(self):
        for idx in range(len(self.camera_arr)):
            self.camera_arr[idx].listen(lambda image, queue=self.img_queue_arr[idx]: process_image(queue, image))

    def destroy(self):
        for camera in self.camera_arr:
            camera.destroy()

    def get_cams_global_loc(self):
        return [(camera.get_transform().location.x, 
                 camera.get_transform().location.y, 
                 camera.get_transform().location.z) for camera in self.camera_arr]
    
    def get_cams_rot_mat(self):
        return [euler_to_rotation_matrix(camera.get_transform().rotation.roll, 
                                         camera.get_transform().rotation.pitch, 
                                         camera.get_transform().rotation.yaw) for camera in self.camera_arr]

    def get_veh_global_loc(self):
        return (self.tar_car.get_transform().location.x, 
                self.tar_car.get_transform().location.y, 
                self.tar_car.get_transform().location.z)
    
    def get_veh_rot_mat(self):
        return euler_to_rotation_matrix(self.tar_car.get_transform().rotation.roll, 
                                        self.tar_car.get_transform().rotation.pitch, 
                                        self.tar_car.get_transform().rotation.yaw)
    
    def get_cam_intrinsics(self):
        self.cam_intrinsics = []
        cam_focus_length = [cam_param[2][0] / (2 * np.tan(cam_param[2][2] * np.pi / 360)) for cam_param in self.cam_params]
        cam_center_x = [cam_param[2][0] / 2 for cam_param in self.cam_params]
        cam_center_y = [cam_param[2][1] / 2 for cam_param in self.cam_params]
        for f, cx, cy in zip(cam_focus_length, cam_center_x, cam_center_y):
            self.cam_intrinsics.append(np.array([[f, 0, cx],
                                                 [0, f, cy],
                                                 [0, 0, 1]]))
        
        return self.cam_intrinsics


if __name__ == '__main__':
    # CAM_WIDTH = 640
    # CAM_HEIGHT = 360
    CAM_WIDTH = 1600
    CAM_HEIGHT = 900

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle.*')[0]
    spawn_points = world.get_map().get_spawn_points()
    actors = world.get_actors()

    for actor in world.get_actors():
        if actor.attributes.get('role_name') == 'hero':
            vehicle = actor
            break
    
    
    cam_bp_70 = cam_bp_maker(blueprint_library, CAM_WIDTH, CAM_HEIGHT, 70)
    cam_bp_110 = cam_bp_maker(blueprint_library, CAM_WIDTH, CAM_HEIGHT, 110)
    

    cameras_params_list = [
        # (x, y, z), (roll, pitch, yaw), (W, H, FOV)
        [(1.3, 0, 1.55), (0, 0, 0), (CAM_WIDTH, CAM_HEIGHT, 70)],    # FRONT
        [(1.3, 1, 1.55), (0, 0, 70), (CAM_WIDTH, CAM_HEIGHT, 70)],   # FRONT_RIGHT
        [(1.3, -1, 1.55), (0, 0, -70), (CAM_WIDTH, CAM_HEIGHT, 70)], # FRONT_LEFT
        
        [(-2.3, 0, 1.55), (0, 0, 180), (CAM_WIDTH, CAM_HEIGHT, 110)],       # BACK
        [(-2.3, -1, 1.55), (0, 0, -140), (CAM_WIDTH, CAM_HEIGHT, 70)],   # BACK_LEFT
        [(-2.3, 1, 1.55), (0, 0, 140), (CAM_WIDTH, CAM_HEIGHT, 70)]    # BACK_RIGHT
        ]

    cam_manager = CameraManager(world, vehicle, cameras_params_list)
    cam_manager.set_cams()
    cam_manager.get_cam_intrinsics()
    cam_manager.listen()

    init_combined_img = np.zeros((CAM_HEIGHT*2, CAM_WIDTH*3, 3), dtype=np.uint8)

    # display_thread = threading.Thread(target=display_images, args=(cam_manager.img_queue_arr, 
    #                                                                init_combined_img, 
    #                                                                CAM_HEIGHT, 
    #                                                                CAM_WIDTH, 
    #                                                                exit_event), daemon=True)
    # display_thread.start()

    send_thread = threading.Thread(target=packed_data_send, args=(cam_manager.img_queue_arr, 
                                                                    init_combined_img, 
                                                                    CAM_HEIGHT, 
                                                                    CAM_WIDTH, 
                                                                    exit_event), daemon=True)
    send_thread.start()

    location_thread = threading.Thread(target=show_location, args=(cam_manager, exit_event), daemon=True)
    location_thread.start()

    try:
        while not exit_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("exit")
    finally:
        exit_event.set()
        cam_manager.destroy()
        cv2.destroyAllWindows()
