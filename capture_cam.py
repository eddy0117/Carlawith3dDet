import carla
import numpy as np
import cv2
import threading
import queue
import time
from utils import euler_to_rotation_matrix, display_images, show_location, process_image

exit_event = threading.Event()





class CameraManager:
    def __init__(self, camera_bp, world):
        self.camera_arr = []
        self.img_queue_arr = []
        self.camera_bp = camera_bp
        self.world = world
    
    def set_cams(self, tar_car, transform_arr):
        for loc in transform_arr:
            self.camera_arr.append(self.world.spawn_actor(self.camera_bp, loc, attach_to=tar_car))
            self.img_queue_arr.append(queue.Queue(maxsize=10))
        return self.camera_arr
    
    def listen(self):
        for idx in range(len(self.camera_arr)):
            self.camera_arr[idx].listen(lambda image, queue=self.img_queue_arr[idx]: process_image(queue, image))

    def destroy(self):
        for camera in self.camera_arr:
            camera.destroy()

    def get_global_loc(self):
        return [(camera.get_transform().location.x, 
                 camera.get_transform().location.y, 
                 camera.get_transform().location.z) for camera in self.camera_arr]
    
    def get_rot_mat(self):
        return [euler_to_rotation_matrix(camera.get_transform().rotation.roll, 
                                         camera.get_transform().rotation.pitch, 
                                         camera.get_transform().rotation.yaw) for camera in self.camera_arr]




if __name__ == '__main__':
    CAM_WIDTH = 640
    CAM_HEIGHT = 360

    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle.*')[0]
    spawn_points = world.get_map().get_spawn_points()
    actors = world.get_actors()

    vehicle = actors.filter('vehicle.*')[0]

    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', f'{CAM_WIDTH}')
    camera_bp.set_attribute('image_size_y', f'{CAM_HEIGHT}')
    camera_bp.set_attribute('fov', '70')
    camera_bp_rear = blueprint_library.find('sensor.camera.rgb')
    camera_bp_rear.set_attribute('image_size_x', f'{CAM_WIDTH}')
    camera_bp_rear.set_attribute('image_size_y', f'{CAM_HEIGHT}')
    camera_bp_rear.set_attribute('fov', '90')
    cameras_transform = [
        carla.Transform(carla.Location(x=1.3, y=-2, z=2.4), carla.Rotation(yaw=-70)), # FRONT_LEFT
        carla.Transform(carla.Location(x=1.3, z=2.4), carla.Rotation(yaw=0)),         # FRONT
        carla.Transform(carla.Location(x=1.3, y=2, z=2.4), carla.Rotation(yaw=70)),   # FRONT_RIGHT
        carla.Transform(carla.Location(x=-3.3, y=-2, z=2.4), carla.Rotation(yaw=-140)),   # BACK_LEFT
        carla.Transform(carla.Location(x=-3.3, z=2.4), carla.Rotation(yaw=180)),       # BACK
        carla.Transform(carla.Location(x=-3.3, y=2, z=2.4), carla.Rotation(yaw=140)), # BACK_RIGHT
                         ]

    cam_manager = CameraManager(camera_bp, world)
    cam_manager.set_cams(vehicle, cameras_transform)
    cam_manager.listen()

    init_combined_img = np.zeros((CAM_HEIGHT*2, CAM_WIDTH*3, 3), dtype=np.uint8)

    display_thread = threading.Thread(target=display_images, args=(cam_manager.img_queue_arr,init_combined_img, CAM_HEIGHT, CAM_WIDTH, exit_event), daemon=True)
    display_thread.start()

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
