import os
import time
import asyncio
from multiprocessing import Pool
import numpy as np
from ultralytics import YOLO
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# 禁用 ultralytics 的日志输出
logging.getLogger("ultralytics").setLevel(logging.WARNING)
def process_images_in_process(image_paths):
    """
    在每个进程中处理分配到的图像列表，利用异步方式提高效率
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    model = YOLO(r"你的YOLO模型，可以是pt，onnx，trt")
    tasks = [async_predict(model, path, name='exp') for path in image_paths]
    results = loop.run_until_complete(asyncio.gather(*tasks))
    loop.close()
    return results

async def async_predict(model, image_path, name='exp'):
    """
    使用异步方式执行单张图像的预测任务
    """
    loop = asyncio.get_running_loop()
    try:
        results = await loop.run_in_executor(None, lambda: model.predict(image_path,
                                                                        batch=1,
                                                                        save=False,
                                                                        conf=0.5,
                                                                        device='cuda',
                                                                        save_crop=True,
                                                                        half=True,
                                                                        save_txt=True,
                                                                        name=name,
                                                                        exist_ok=True))
        return results
    except Exception as e:
        logging.error(f"Error predicting image {image_path}: {e}")
        return None

def distribute_images_to_processes(folder_path, num_processes):
    """
    读取文件夹下所有图像文件，并分配给指定数量的图像文件
    """
    image_extensions = (".jpg", ".png", ".jpeg", ".bmp")
    image_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(image_extensions)]
    all_image_paths = np.array_split(image_files, num_processes)
    with Pool(num_processes) as pool:
        results = pool.map(process_images_in_process, all_image_paths)
    return results

if __name__ == "__main__":
    folder_path = r"你的图像路径"
    num_processes = 16  # 根据实际情况调整进程数量，结合GPU性能优化
    #16=81s
    start_time = time.time()
    final_results = distribute_images_to_processes(folder_path, num_processes)
    end_time = time.time()
    logging.info(f"Total time taken: {end_time - start_time} seconds")
