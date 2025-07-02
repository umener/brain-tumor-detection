#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :logging_utils.py
# @Time      :2025/6/23 14:28:17
# @Author    :雨霓同学
# @Project   :BTD
# @Function  :日志相关的工具类函数
import logging
from datetime import datetime
from pathlib import Path


def setup_logger(base_path: Path, log_type: str = "general",
                model_name: str = None,
                encoding: str = "utf-8",
                log_level: int = logging.INFO,
                temp_log: bool = False,
                logger_name: str = "YOLO Default"
                ):
    """
    配置日志记录器，将日志保存到指定路径的子目录当中，并同时输出到控制台，日志文件名为类型 + 时间戳
    :param model_name: 模型训练可能需要一个模型的名字，我们可以传入日志记录器，生成带模型名的日志文件
    :param log_type: 日志的类型
    :param base_path: 日志文件的根路径
    :param encoding: 文件编码
    :param log_level: 日志等级
    :param temp_log: 是否启动临时文件名
    :param logger_name: 日志记录器的名称
    :return: logging.logger: 返回一个日志记录器实例
    """
    # 1. 构建日志文件完整的存放路径
    log_dir = base_path / log_type
    log_dir.mkdir(parents=True, exist_ok=True)

    # 2. 生成一个带时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    # 根据temp_log参数，生成不同的日志文件名
    prefix = "temp_" if temp_log else log_type.replace(" ", "-")
    log_filename_parts = [prefix, timestamp]
    if model_name:
        log_filename_parts.append(model_name.replace(" ", "-"))
    log_filename = "_".join(log_filename_parts) + ".log"
    log_file = log_dir / log_filename

    # 3. 获取或创建指定的名称logger实例
    logger = logging.getLogger(logger_name)
    # 设定日志记录器记录最低记录级别
    logger.setLevel(log_level)
    # 阻止日志事件传播到父级logger
    logger.propagate = False

    # 4. 需要避免重复添加日志处理器，因此先检查日志处理器列表中是否已经存在了指定的日志处理器
    if logger.handlers:
        for handler in logger.handlers:
            logger.removeHandler(handler)
    # 5.创建文件处理器，将日志写入到文件当中
    file_handler = logging.FileHandler(log_file, encoding=encoding)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s : %(message)s"))
    # 将文件处理器添加到logger实例中
    logger.addHandler(file_handler)

    # 6.创建控制台处理器，将日志输出到控制台
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s : %(message)s"))
    # 将控制台处理器添加到logger实例中
    logger.addHandler(console_handler)

    # 输出一些初始化信息到日志，确认配置成功
    logger.info(f"日志记录器初始化开始".center(60, "="))
    logger.info(f"当前日志记录器的根目录: {base_path}")
    logger.info(f"当前日志记录器的名称: {logger_name}")
    logger.info(f"当前日志记录器的类型: {log_type}")
    logger.info(f"单前日志记录器的级别: {logging.getLevelName(log_level)}")
    logger.info("日志记录器初始化成功".center(60, "="))
    return logger


def rename_log_file(logger_obj, save_dir, model_name, encoding="utf-8"):
    """
    主要实现日志的重命名,如train1, train2, train3....
    :param logger_obj:
    :param save_dir:
    :param model_name:
    :param encoding:
    :return:
    """
    # 遍历当前日志记录器
    for handler in list(logger_obj.handlers):
        if isinstance(handler, logging.FileHandler):
            old_log_file = Path(handler.baseFilename)
            timestamp_parts = old_log_file.stem.split("_")
            timestamp = timestamp_parts[2]
            train_prefix = Path(save_dir).name
            new_log_file = old_log_file.parent / f"{train_prefix}_{timestamp}_{model_name}.log"

            # 关闭旧的日志处理器
            handler.close()

            logger_obj.removeHandler(handler)

            if old_log_file.exists():
                try:
                    old_log_file.rename(new_log_file)
                    logger_obj.info(f"日志文件已经重命名成功: {new_log_file}")
                except OSError as e:
                    logger_obj.error(f"日志文件重命名失败: {e}")
                    re_added_handler = logging.FileHandler(old_log_file,encoding='utf-8')
                    re_added_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
                    logger_obj.addHandler(re_added_handler)
                    continue
            else:
                logger_obj.warning(f"尝试重命名的日志文件不存在: {old_log_file}")

            # 命名成功处理方案
            new_handler = logging.FileHandler(new_log_file,encoding='utf-8')
            new_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
            logger_obj.addHandler(new_handler)
            break

def log_parameters(args, exclude_params=None, logger=None):
    """
    记录命令行和YAML参数信息，返回结构化字典。

    Args:
        args: 命令行参数（Namespace 对象）
        exclude_params: 不记录的参数键列表
        logger: 日志记录器实例

    Returns:
        dict: 参数字典
    """
    if logger is None:
        logger = logging.getLogger("YOLO_Training")
    if exclude_params is None:
        exclude_params = ['log_encoding', 'use_yaml', 'log_level', 'extra_args']
    logger.info("开始模型参数信息".center(40, '='))
    logger.info("Parameters")
    logger.info("-" * 40)
    params_dict = {}
    for key, value in vars(args).items():
        if key not in exclude_params and not key.endswith('_specified'):
            source = '命令行' if getattr(args, f'{key}_specified', False) else 'YAML'
            logger.info(f"{key:<20}: {value}  来源: [{source}]")
            params_dict[key] = {"value": value, "source": source}
    return params_dict



if __name__ == "__main__":
    from paths import LOGS_DIR
    logger_ = setup_logger(base_path=LOGS_DIR,
                        log_type="test_log", model_name=None,
                        )
    logger_.info("测试日志记录器")
