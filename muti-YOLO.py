import os
import time
import asyncio
from multiprocessing import Pool
import numpy as np
from ultralytics import YOLO
import logging
import uuid
import cv2

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("ultralytics").setLevel(logging.WARNING)

def process_images_in_process(image_paths):
    """
    在每个进程中处理分配到的图像列表，利用异步方式提高效率
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    model = YOLO(r"D:\PYCHARMproject\TFDS_v8\ultralytics\runs\detect_12-30before\TFDSALLyolo11\weights\best.engine")
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
        # 禁用自动保存裁剪图像和标签文件
        results = await loop.run_in_executor(None, lambda: model.predict(image_path,
                                                                        batch=1,
                                                                        save=False,  # 禁用自动保存
                                                                        conf=0.5,
                                                                        device='cuda',
                                                                        save_crop=False,  # 禁用自动保存裁剪图像
                                                                        half=True,
                                                                        save_txt=False,  # 禁用自动保存标签文件
                                                                        name=name,
                                                                        exist_ok=True))
        
        # 获取保存路径
        save_dir = os.path.join("runs", name)
        os.makedirs(os.path.join(save_dir, "crops"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "labels"), exist_ok=True)
        
        # 获取图像文件名（不含扩展名）
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # 遍历检测结果
        for i, result in enumerate(results):
            # 保存裁剪图像
            if result.boxes is not None and len(result.boxes) > 0:
                for j, box in enumerate(result.boxes):
                    # 获取类别名称
                    class_name = model.names[int(box.cls)]  # 根据类别 ID 获取类别名称
                    class_dir = os.path.join(save_dir, "crops", class_name)
                    os.makedirs(class_dir, exist_ok=True)
                    
                    # 裁剪图像
                    crop = result.orig_img[int(box.xyxy[0][1]):int(box.xyxy[0][3]),
                                          int(box.xyxy[0][0]):int(box.xyxy[0][2])]
                    
                    # 生成唯一文件名
                    unique_id = uuid.uuid4().hex[:8]  # 生成 8 位唯一字符串
                    crop_filename = f"{image_name}_{j}_{unique_id}.jpg"  # 保持原有格式，添加唯一标识符
                    cv2.imwrite(os.path.join(class_dir, crop_filename), crop)
            
            # 保存标签文件
            if result.boxes is not None and len(result.boxes) > 0:
                label_filename = f"{image_name}_{i}_{uuid.uuid4().hex[:8]}.txt"
                with open(os.path.join(save_dir, "labels", label_filename), "w") as f:
                    for box in result.boxes:
                        f.write(f"{box.cls} {box.xywhn[0][0]} {box.xywhn[0][1]} {box.xywhn[0][2]} {box.xywhn[0][3]}\n")
        
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
    folder_path = r"D:\PYCHARMproject\TFDS\3178VIEW"
    num_processes = 16  # 根据实际情况调整进程数量，结合GPU性能优化
    start_time = time.time()
    final_results = distribute_images_to_processes(folder_path, num_processes)
    end_time = time.time()
    logging.info(f"Total time taken: {end_time - start_time} seconds")
