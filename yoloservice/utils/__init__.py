#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :__init__.py.py
# @Time      :2025/6/23 13:44:36
# @Author    :雨霓同学
# @Project   :BTD
# @Function  :
# from utils import setup_logger
from .logging_utils import setup_logger
from .performance_utils import time_it
from .paths import (
    YOLOSERVER_ROOT,  # 项目根目录
    CONFIGS_DIR,  # 配置文件目录
    DATA_DIR,  # 数据集
    RUNS_DIR,   # 模型运行结果 目录
    LOGS_DIR,   # 日志目录
    MODEL_DIR,
    PRETRAINED_DIR,  # 预训练模型存放的位置
    CHECKPOINTS_DIR,
    RAW_DATA_DIR,
    RAW_IMAGES_DIR,
    ORIGINAL_ANNOTATIONS_DIR,
    YOLO_STAGED_LABELS_DIR,
    SCRIPTS_DIR
)