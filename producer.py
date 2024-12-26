import numpy as np
from multiprocessing import shared_memory
import time

IMG_COUNT = 6
IMG_SHAPE = (900, 1600, 3)  # 圖像形狀 (H, W, C)
SHARED_MEM_NAME = "image_data"
FLAG_MEM_NAME = "flag_data"

if __name__ == "__main__":
    # 創建共享內存
    shm = shared_memory.SharedMemory(create=True, name=SHARED_MEM_NAME, size=IMG_COUNT * np.prod(IMG_SHAPE))
    flag_shm = shared_memory.SharedMemory(create=True, name=FLAG_MEM_NAME, size=IMG_COUNT)

    # 將共享內存映射為 Numpy 數組
    img_array = np.ndarray((IMG_COUNT, *IMG_SHAPE), dtype=np.uint8, buffer=shm.buf)
    flags = np.ndarray((IMG_COUNT,), dtype=np.uint8, buffer=flag_shm.buf)

    try:
        for i in range(IMG_COUNT):
            # 隨機生成圖像
            img_array[i] = np.random.randint(0, 255, IMG_SHAPE, dtype=np.uint8)
            flags[i] = 1  # 標記該圖像已就緒
            print(f"Producer: Generated image {i}")
            time.sleep(1)  # 模擬數據生成延遲
    finally:
        shm.close()
        flag_shm.close()
