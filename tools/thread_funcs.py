import time
import socket
import base64
import threading
import queue
import json

import carla
import numpy as np
import cv2



MAX_CHUNK_SIZE = 50000
ip = "127.0.0.1"
port = 65432
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]

def packed_data_send(camera_manager, CAM_HEIGHT, CAM_WIDTH, exit_event):
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((ip, port))
    img_arr = [np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8) for _ in range(6)]

    while not exit_event.is_set():
        other_params = {'c2g_t':    list(camera_manager.get_cams_global_loc()), 
                        'c2g_r_m':  np.array(camera_manager.get_cams_global_rot_mat()).tolist(),
                        'e2g_t': list(camera_manager.get_ego_global_loc()),
                        'e2g_r_m': camera_manager.get_ego_global_rot_mat().tolist()}
        encoded_dict = json.dumps(other_params).encode("utf-8")
        data_send = encoded_dict
        data_send += ("\1").encode("utf-8")
        # 在同一時間點可能有些 queue 是空的，所以使用 list 存放每個 queue 的 image
        # 這樣就算當前 queue 是空的，最後 encode 時也還會是上一個 frame 的 image
        for idx, image_queue in enumerate(camera_manager.img_queue_arr):
            if not image_queue.empty():
                image = image_queue.get()
                array = np.frombuffer(image.raw_data, dtype=np.uint8)
                array = array.reshape((image.height, image.width, 4))  # BGRA
                rgb_image = array[:, :, :3]  # BGR
                img_arr[idx] = rgb_image

        
        for idx, rgb_image in enumerate(img_arr):
            img_encoded = base64.b64encode(cv2.imencode('.jpg', rgb_image, encode_param)[1]).decode("utf-8")
            data_send += img_encoded.encode("utf-8")
            data_send += ("\1").encode("utf-8")
            
        data_send += ("\0").encode("utf-8")
        for i in range(0, len(data_send), MAX_CHUNK_SIZE):
            client_socket.sendall(data_send[i : i + MAX_CHUNK_SIZE])
        time.sleep(0.03)

def display_images(camera_queues, combined_img, CAM_HEIGHT, CAM_WIDTH, exit_event):
    display_order = [1, 2, 0, 4, 3, 5]
    while not exit_event.is_set():
        for idx, image_queue in enumerate(camera_queues):
            if not image_queue.empty():
                image = image_queue.get()
                array = np.frombuffer(image.raw_data, dtype=np.uint8)
                array = array.reshape((image.height, image.width, 4))  # BGRA
                rgb_image = array[:, :, :3]  # BGR
                col_idx = display_order[idx] // 3
                row_idx = display_order[idx] % 3
                combined_img[CAM_HEIGHT*col_idx:CAM_HEIGHT*(col_idx+1), CAM_WIDTH*row_idx:CAM_WIDTH*(row_idx+1)] = rgb_image
        cv2.imshow(f"Camera RGB_{idx}", combined_img)
        if cv2.waitKey(1) == ord('q'):
            exit_event.set()
            break
    cv2.destroyAllWindows()

def show_location(camera_manager, exit_event):
    while not exit_event.is_set():
        time.sleep(0.5)
        print('vehicle loc:', [round(loc, 3) for loc in camera_manager.get_cams_global_loc()[0]])
        print('vehicle rot:', camera_manager.get_cams_global_rot_mat()[0])
  

def process_image(image_queue, image):
    if not image_queue.full():
        image_queue.put(image)

