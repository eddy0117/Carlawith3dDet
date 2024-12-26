import time
import json
import socket
import numpy as np
import base64
import cv2
import os


# Set the maximum size of the message that can be transmited one time
MAX_CHUNK_SIZE = 50000
ip = "127.0.0.1"
port = 65432
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]

if __name__ == "__main__":
    
    
    # Create a UDP socket
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((ip, port))

    img_paths = os.listdir("fake_imgs")

    for _ in range(6):
        data_send = b""

        fake_dict = {'speed': 0.0, 'location': [0.0, 0.0, 0.0], 'rotation': [0.0, 0.0, 0.0]}
        encoded_dict = json.dumps(fake_dict).encode("utf-8")
        data_send += encoded_dict
        data_send += ("\1").encode("utf-8")

        for path in img_paths:
            fake_img = cv2.imread(os.path.join("fake_imgs", path))
            img_encoded = base64.b64encode(cv2.imencode('.jpg', fake_img, encode_param)[1]).decode("utf-8")
            data_send += img_encoded.encode("utf-8")
            data_send += ("\1").encode("utf-8")
        data_send += ("\0").encode("utf-8")

        print(len(data_send))

        for i in range(0, len(data_send), MAX_CHUNK_SIZE):
            client_socket.sendall(data_send[i : i + MAX_CHUNK_SIZE])

        time.sleep(0.3)
