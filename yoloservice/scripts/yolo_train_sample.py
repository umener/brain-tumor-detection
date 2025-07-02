#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :yolo_train_sample.py
# @Time      :2025/6/25 16:03:38
# @Author    :雨霓同学
# @Project   :BTD
# @Function  :
from ultralytics import YOLO

def simple_yolo_train():
    model = YOLO(r'yolo11n-seg.pt')
    results = model.train(
        data=r'D:\BTD\yoloservice\configs\data.yaml',  # 假设data.yaml在当前运行目录下
        epochs=200,  # 训练50个epochs
        imgsz=640,  # 图像大小为640x640
        batch=16,  # 自动选择合适的batch size
        device='0',
        name='yolo11mn-seg'  # 训练结果保存目录的名称
    )

    print(results)
    print("模型存放点",model.trainer.save_dir) # 模型存放点


if __name__ == '__main__':
    simple_yolo_train()

