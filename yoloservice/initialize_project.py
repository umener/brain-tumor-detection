#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :initialize_project.py
# @Time      :2025/6/24 09:09:24
# @Author    :雨霓同学
# @Project   :BTD
# @Function  :项目初始化脚本，检查并创建必要的项目结构，提示用户将原始数据存放到指定的位置

import logging

from utils import setup_logger
from utils import time_it
from utils import (
    YOLOSERVER_ROOT,  # 项目根目录
    CONFIGS_DIR,  # 配置文件目录
    DATA_DIR,  # 数据集
    RUNS_DIR,  # 模型运行结果 目录
    LOGS_DIR,  # 日志目录
    MODEL_DIR,  #
    PRETRAINED_DIR,  # 预训练模型存放的位置
    CHECKPOINTS_DIR,
    SCRIPTS_DIR,
    RAW_IMAGES_DIR,
    ORIGINAL_ANNOTATIONS_DIR,
    YOLO_STAGED_LABELS_DIR,
)

# 第一步：配置日志记录
logger = setup_logger(base_path=LOGS_DIR,
                        log_type="init_project",
                        model_name=None,
                        log_level=logging.INFO,
                        logger_name="YOLO Initialize Project"
                        )


# 第二步：定义项目初始化函数
@time_it(iterations=1, name="项目初始化",logger_instance=logger)
def initialize_project():
    """
    检查并创建项目所需的文件夹结构
    :return:
    """
    logger.info("开始初始化项目".center(60, "="))
    logger.info(f"当前项目的根目录为：{YOLOSERVER_ROOT.resolve()}")
    created_dirs = []
    existing_dirs = []
    raw_data_status = []

    standard_data_to_create = [
        CONFIGS_DIR,
        DATA_DIR,
        RUNS_DIR,
        MODEL_DIR,
        CHECKPOINTS_DIR,
        PRETRAINED_DIR,
        LOGS_DIR,
        SCRIPTS_DIR,
        DATA_DIR / "train" / "images",
        DATA_DIR / "val" / "images",
        DATA_DIR / "test" / "images",
        DATA_DIR / "train" / "labels",
        DATA_DIR / "val" / "labels",
        DATA_DIR / "test" / "labels",
        YOLO_STAGED_LABELS_DIR,
        ORIGINAL_ANNOTATIONS_DIR,
    ]

    logger.info(f"检查并创建核心项目目录结构".center(80, "="))
    for d in standard_data_to_create:
        if not d.exists():
            try:
                d.mkdir(parents=True, exist_ok=True)
                logger.info(f" 已经创建的目录：{d.relative_to(YOLOSERVER_ROOT)}")
                created_dirs.append(d)
            except Exception as e:
                logger.error(f" 创建目录：{d.relative_to(YOLOSERVER_ROOT)} 失败: {e}")
                created_dirs.append(f" 创建目录：{d.relative_to(YOLOSERVER_ROOT)} 失败: {e}")
        else:
            logger.info(f" 检测到已存在的目录：{d.relative_to(YOLOSERVER_ROOT)}")
            existing_dirs.append(d.relative_to(YOLOSERVER_ROOT))
    logger.info(f"核心项目文件夹结构检查以及创建完成".center(60, "="))

    # 3. 检查原始数据集目录并给出提示
    logger.info(f"开始检查原始数据集目录".center(60, "="))
    raw_dirs_to_check = {
        "原始图像文件": RAW_IMAGES_DIR,
        "原始标注文件": ORIGINAL_ANNOTATIONS_DIR,
    }
    for desc, raw_dir in raw_dirs_to_check.items():
        if not raw_dir.exists():
            msg = (
                f"!! 原始{desc}目录不存在，请将原始数据集数据放置此目录下，"
                f"并确保目录结构正确，以便后续数据集转换正常执行"
            )
            logger.warning(msg)
            logger.warning(f"期望结构为: {raw_dir.resolve()}")
            raw_data_status.append(f"{raw_dir.relative_to(YOLOSERVER_ROOT)}:不存在，需要手动创建并放置原始数据")
        else:
            if not any(raw_dir.iterdir()):
                msg = f"原始{desc}，已经存在，但内容为空，请将原始{desc}放在此目录下，以便后续数据集转换"
                logger.warning(msg)
                raw_data_status.append(f"{raw_dir.relative_to(YOLOSERVER_ROOT)}:已经存在，但内容为空,需要放置原始数据")
            else:
                logger.info(f"原始{desc}，已经存在, {raw_dir.relative_to(YOLOSERVER_ROOT)}包含原始文件")
                raw_data_status.append(f"{raw_dir.relative_to(YOLOSERVER_ROOT)}:已经存在")

    # 第四步：汇总所有的检查结果和创建结果
    logger.info("项目初始化结果汇总".center(80, "="))
    if created_dirs:
        logger.info(f"此次初始化过程中,一共创建了 {len(created_dirs)}个目录，具体内容如下：")
        for d in created_dirs:
            logger.info(f"- {d}")
    else:
        logger.info("本次初始化没有创建任何目录")

    if existing_dirs:
        logger.info(f"此次初始化过程中，一共检查到 {len(existing_dirs)}个 已经存在的目录,具体内容如下:")
        for d in existing_dirs:
            logger.info(f"- {d}")
    if raw_data_status:
        logger.info(f"原始数据集目录检查结果".center(80,'='))
        for s in raw_data_status:
            logger.info(f"- {s}")
    logger.info("请务必根据上述提示进行操作，特别是关于原始数据集目录的检查结果")
    logger.info("初始化项目完成".center(80,'='))

if __name__ == '__main__':
    initialize_project()

