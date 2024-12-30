import json
import socket
import base64
import cv2
import numpy as np




# Set the maximum size of the message that can be received one time
MAX_CHUNK_SIZE = 50000
ip = "127.0.0.1"
port = 65432


if __name__ == "__main__":
    while True:
        # Create a TCP socket
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Bind the socket to the address and port
        server_socket.bind((ip, port))
        server_socket.listen(1)

        conn, _ = server_socket.accept()

        whole_data = b""
        data_cat = b""
        while True:
            data = conn.recv(MAX_CHUNK_SIZE)
            data = data_cat + data  # add the rest of the data from last frame

            if not data:
                server_socket.close()
                print("Connection closed")
                break

            data_split = data.split(b"\0")
        
            if len(data_split) > 1:  # End of a package
                data_cat = data_split[1]  # Preserve the rest of the data (a part of next frame first chunk)
                whole_data += data_split[0]
    
                img_data = whole_data.split(b"\1")[1:-1]
                json_data = whole_data.split(b"\1")[0]
             

                for i, single_img_data in enumerate(img_data):
                    single_img_data = base64.b64decode(single_img_data)
                    single_img = np.frombuffer(single_img_data, dtype=np.uint8)
                    single_img = cv2.imdecode(single_img, cv2.IMREAD_COLOR)
                    
                    cv2.imwrite(f"Image_{i}.jpg", single_img)
       
                json_data = json.loads(json_data.decode("utf-8"))
                print(json_data)

                # dict_data = json.loads(whole_data.split(b"\1")[0].decode("utf-8"))
                # print(dict_data)
                whole_data = b""

            else:
                data_cat = b""
                whole_data += data_split[0]
