import carla
import numpy as np
import cv2
import threading
import queue
import time



def euler_to_rotation_matrix(roll, pitch, yaw):
    # Convert angles to radians if they are in degrees
    roll = np.radians(roll)
    pitch = np.radians(pitch)
    yaw = np.radians(yaw)
    
    # Rotation matrices
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    R_y = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    R_z = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Final rotation matrix (adjust order if needed)
    R = R_z @ R_y @ R_x
    return R

def display_images(camera_queues, combined_img, CAM_HEIGHT, CAM_WIDTH, exit_event):
    while not exit_event.is_set():
        
        for idx, image_queue in enumerate(camera_queues):
            if not image_queue.empty():
                image = image_queue.get()
                array = np.frombuffer(image.raw_data, dtype=np.uint8)
                array = array.reshape((image.height, image.width, 4))  # BGRA
                rgb_image = array[:, :, :3]  # BGR
                col_idx = idx // 3
                row_idx = idx % 3
                combined_img[CAM_HEIGHT*col_idx:CAM_HEIGHT*(col_idx+1), CAM_WIDTH*row_idx:CAM_WIDTH*(row_idx+1)] = rgb_image
        cv2.imshow(f"Camera RGB_{idx}", combined_img)
        if cv2.waitKey(1) == ord('q'):
            exit_event.set()
            break
    cv2.destroyAllWindows()

def show_location(camera_manager, exit_event):
    while not exit_event.is_set():
        time.sleep(0.5)
        print('vehicle loc:', [round(loc, 3) for loc in camera_manager.get_global_loc()[0]])
        print('vehicle rot:', camera_manager.get_rot_mat()[0])

def process_image(image_queue, image):
    if not image_queue.full():
        image_queue.put(image)