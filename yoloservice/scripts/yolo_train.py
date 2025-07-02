#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :yolo_train.py
# @Time      :2025/6/26 13:00:05
# @Author    :雨霓同学
# @Project   :BTD
# @Function  :训练脚本的入口,集成utils模块
import logging

from ultralytics import YOLO
import argparse
from pathlib import Path
import sys

current_path = Path(__file__).parent.parent.resolve()
utils_path = current_path / 'utils'
if str(current_path) not in sys.path:
    sys.path.insert(0, str(current_path))
if str(utils_path) not in sys.path:
    sys.path.insert(1, str(utils_path))

from utils import setup_logger, time_it
from utils.paths import LOGS_DIR, CHECKPOINTS_DIR, PRETRAINED_DIR
from utils.config_utils import load_config, merge_config
from utils.system_utils import log_device_info
from utils.logging_utils import rename_log_file
from utils.model_utils import copy_checkpoint_models
from utils.result_utils import log_results
from utils.datainfo_utils import log_dataset_info
from utils.logging_utils import log_parameters
def parse_args():
    """命名行解析参数"""
    parser = argparse.ArgumentParser(description="YOLO 模型训练")
    parser.add_argument("--data", type=str, default="data.yaml",help="yaml配置文件路径")
    parser.add_argument("--batch", type=int, default=16,help="训练批次大小")
    parser.add_argument("--epochs", type=int, default=200,help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=640,help="训练图片尺寸")
    parser.add_argument("--device", type=str, default="0",help="训练设备")
    parser.add_argument("--weights", type=str, default= "yolo11n.pt",help="预训练模型路径")
    parser.add_argument("--workers",type=int, default=8,help="训练数据加载线程数" )
    # 自定义参数
    parser.add_argument("--use_yaml", type=bool, default=True,help="使用yaml配置文件" )
    return parser.parse_args()

def run_training(model, yolo_args):
    result = model.train(**vars(yolo_args))
    return result

def main(args, logger):
    logger.info("YOLO 肿瘤检测训练脚本启动".center(80, "="))
    try:
        yaml_config = {}
        if args.use_yaml:
            yaml_config = load_config(config_type='train')

        # 合并参数
        yolo_args, project_args = merge_config(args,yaml_config, mode='train')

        # 记录设备信息
        log_device_info(logger)

        # 获取数据信息
        log_dataset_info(args.data, mode='train', logger=logger)

        # 记录参数来源
        log_parameters(project_args, logger=logger)

        # 初始化模式,开始执行训练
        logger.info(f"初始化模型,加载模型: {project_args.weights}")
        model_path = PRETRAINED_DIR / project_args.weights
        if not model_path.exists():
            logger.info(f"模型文件不存在: {model_path},请将{project_args.weights}放入到{PRETRAINED_DIR}")
            raise ValueError(f"模型文件不存在: {model_path}")
        model = YOLO(model_path)


        # 动态引用time_it装饰器
        decorated_run_training = time_it(iterations=1, name="模型训练", logger_instance=logger)(run_training)
        results = decorated_run_training(model, yolo_args)

        # 记录结果信息
        log_results(results, logger, model.trainer)

        model_name_for_log = project_args.weights.replace(".pt", "")
        rename_log_file(logger, str(model.trainer.save_dir), model_name_for_log)

        copy_checkpoint_models(Path(model.trainer.save_dir),
                            project_args.weights,
                            CHECKPOINTS_DIR, logger)

        logger.info(f"YOLO 肿瘤检测训练脚本结束")
    except Exception as e:
        logger.error(f"参数合并失败: {e}")
        return

if __name__ == "__main__":
    args_ = parse_args()
    logger = setup_logger(base_path=LOGS_DIR,
                        log_type="train",
                        model_name=args_.weights.replace(".pt", ""),
                        log_level=logging.INFO,
                        temp_log=True,
                        logger_name="YOLO_Training"
                        )
    main(args_, logger)

