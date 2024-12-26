import numpy as np
from multiprocessing import shared_memory
import cv2
import time

IMG_COUNT = 6
IMG_SHAPE = (900, 1600, 3)  # 圖像形狀 (H, W, C)
SHARED_MEM_NAME = "image_data"
FLAG_MEM_NAME = "flag_data"

if __name__ == "__main__":
    # 連接到共享內存
    existing_shm = shared_memory.SharedMemory(name=SHARED_MEM_NAME)
    flag_shm = shared_memory.SharedMemory(name=FLAG_MEM_NAME)

    # 將共享內存映射為 Numpy 數組
    img_array = np.ndarray((IMG_COUNT, *IMG_SHAPE), dtype=np.uint8, buffer=existing_shm.buf)
    flags = np.ndarray((IMG_COUNT,), dtype=np.uint8, buffer=flag_shm.buf)

    try:
        for i in range(IMG_COUNT):
            while flags[i] == 0:
                time.sleep(0.1)  # 等待數據就緒

            # 讀取圖像
            img = img_array[i]
            print(f"Consumer: Received image {i}, shape = {img.shape}")

            # 顯示圖像
            cv2.imshow(f"Image {i}", img)
            cv2.waitKey(500)

            # 清理標記
            flags[i] = 0
    finally:
        existing_shm.close()
        flag_shm.close()
        cv2.destroyAllWindows()

    # 清理共享內存
    shared_memory.SharedMemory(name=SHARED_MEM_NAME).unlink()
    shared_memory.SharedMemory(name=FLAG_MEM_NAME).unlink()
