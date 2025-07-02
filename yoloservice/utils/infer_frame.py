#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :infer_frame.py
# @Time      :2025/6/17 11:40:07
# @Author    :雨霓同学
# @Project   :SafeH
# @Function  :单帧图像美化封装

import cv2
import numpy as np
from utils.beautify import custom_plot # 确保 custom_plot 可以被访问

def process_frame(
    frame,
    result,
    project_args,
    beautify_params,
):
    """
    处理单帧图像，进行绘制。
    此函数现在只进行绘制，不负责最终的显示缩放。它返回原始尺寸的标注图。
    """
    annotated_frame = frame.copy()
    original_height, original_width = frame.shape[:2]

    # --- 确保在这里提取 boxes, confs, labels ---
    # 无论是否使用 custom_plot，这些数据都是从 result 对象中获取的
    # 而且 custom_plot 需要这些作为独立参数
    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    labels_idx = result.boxes.cls.cpu().numpy().astype(int)
    labels = [result.names[int(cls_idx)] for cls_idx in labels_idx]
    # --- 提取结束 ---

    if project_args.beautify:
        # 使用美化函数绘制
        annotated_frame = custom_plot(
            annotated_frame,
            boxes, # 此时 boxes 已经定义
            confs,
            labels,
            use_chinese_mapping=beautify_params.get("use_chinese_mapping", True),
            font_path=beautify_params.get("font_path"),
            font_size=beautify_params.get("font_size", 26),
            line_width=beautify_params.get("line_width",4),
            label_padding_x=beautify_params.get("label_padding_x", 10),
            label_padding_y=beautify_params.get("label_padding_y", 10),
            radius=beautify_params.get("radius", 10),
            text_color_bgr=beautify_params.get("text_color_bgr", (0,0, 0))
        )
    else:
        # 如果不美化，使用YOLO默认的plot函数
        # result.plot() 内部会处理 boxes, confs, labels 的提取
        annotated_frame = result.plot()

    return annotated_frame