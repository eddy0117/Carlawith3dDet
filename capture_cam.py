import carla
import numpy as np
import cv2
import threading
import queue
import time

exit_event = threading.Event()

def process_image(image_queue, image):
    if not image_queue.full():
        image_queue.put(image)

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

def display_images(camera_queues):
    while not exit_event.is_set():
        for idx, image_queue in enumerate(camera_queues):
            if not image_queue.empty():
                image = image_queue.get()
                array = np.frombuffer(image.raw_data, dtype=np.uint8)
                array = array.reshape((image.height, image.width, 4))  # BGRA
                rgb_image = array[:, :, :3]  # BGR
                cv2.imshow(f"Camera RGB_{idx}", rgb_image)
        if cv2.waitKey(1) == ord('q'):
            exit_event.set()
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    client = carla.Client('localhost', 2000)
    client.set_timeout(10.0)
    world = client.get_world()

    blueprint_library = world.get_blueprint_library()
    vehicle_bp = blueprint_library.filter('vehicle.*')[0]
    spawn_points = world.get_map().get_spawn_points()
    actors = world.get_actors()

    vehicle = actors.filter('vehicle.*')[0]

    camera_bp = blueprint_library.find('sensor.camera.rgb')
    camera_bp.set_attribute('image_size_x', '800')
    camera_bp.set_attribute('image_size_y', '600')
    camera_bp.set_attribute('fov', '90')
    cameras_transform = [carla.Transform(carla.Location(x=1.3, z=2.4)), 
                         carla.Transform(carla.Location(x=1.3, y=2, z=2.4), carla.Rotation(yaw=70)),
                         carla.Transform(carla.Location(x=1.3, y=-2, z=2.4), carla.Rotation(yaw=-70))]

    cam_manager = CameraManager(camera_bp, world)
    cam_manager.set_cams(vehicle, cameras_transform)
    cam_manager.listen()


    display_thread = threading.Thread(target=display_images, args=(cam_manager.img_queue_arr,), daemon=True)
    display_thread.start()

    try:
        while not exit_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("exit")
    finally:
        exit_event.set()
        cam_manager.destroy()
        cv2.destroyAllWindows()
