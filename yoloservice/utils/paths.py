#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :paths.py
# @Time      :2025/6/23 13:46:29
# @Author    :雨霓同学
# @Project   :BTD
# @Function  :定义所有的路径信息
from pathlib import Path

# 项目根目录
YOLOSERVER_ROOT = Path(__file__).resolve().parents[1]

# 配置文件目录
CONFIGS_DIR = YOLOSERVER_ROOT / "configs"

# 模型目录
MODEL_DIR = YOLOSERVER_ROOT / "models"

# 训练好的模型存放的位置
CHECKPOINTS_DIR = MODEL_DIR  / "checkpoints"
# 预训练模型存放的位置
PRETRAINED_DIR = MODEL_DIR / "pretrained"


# 模型运行结果 目录
RUNS_DIR = YOLOSERVER_ROOT / "runs"

# 数据文件目录
DATA_DIR = YOLOSERVER_ROOT / "data"

# 原始数据文件目录
RAW_DATA_DIR = DATA_DIR / "raw"

# 原始图像存放目录
RAW_IMAGES_DIR = RAW_DATA_DIR / "images"

# 原始非yolo格式标注文件存放目录
ORIGINAL_ANNOTATIONS_DIR = RAW_DATA_DIR / "original_annotations"

# YOLO格式标注文件暂时存放目录
YOLO_STAGED_LABELS_DIR = RAW_DATA_DIR / "yolo_staged_labels"

# 临时文件存放目录
RAW_TEMP_DIR = RAW_DATA_DIR / "temp"

# 训练验证测试数据集存放目录
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TEST_DIR = DATA_DIR / "test"

# 日志目录
LOGS_DIR = YOLOSERVER_ROOT / "logs"

# 训练推理脚本存放目录
SCRIPTS_DIR = YOLOSERVER_ROOT / "scripts"

if __name__ == "__main__":
    # 目录自动创建
    for _path in [
        YOLOSERVER_ROOT,
        CONFIGS_DIR,
        MODEL_DIR,
        RUNS_DIR,
        CHECKPOINTS_DIR,
        PRETRAINED_DIR,
        DATA_DIR,
        RAW_DATA_DIR,
        RAW_IMAGES_DIR,
        ORIGINAL_ANNOTATIONS_DIR,
        YOLO_STAGED_LABELS_DIR,
        RAW_TEMP_DIR,
        TRAIN_DIR,
        VAL_DIR,
        TEST_DIR,
        LOGS_DIR,
        SCRIPTS_DIR,
    ]:
        _path.mkdir(parents=True, exist_ok=True)

