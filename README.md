# Muti-YOLO-Predict
一个yolo基于TRT多进程处理图像推理的脚本
1.基于Ultralytics库export的TRT模型，batch=1，接受1，3，640，640序列
2.适用于好几千张图像的批量检测，实测从原先的200s到80s
3.根据自己的cpu核心去设置进程数量，不要太多，根据资源调整
4.主要采用了多进程和异步编程
