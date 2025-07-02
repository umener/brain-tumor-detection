#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :config_utils.py
# @Time      :2025/6/26 09:22:17
# @Author    :雨霓同学
# @Project   :BTD
# @Function  :加载配置文件，生成默认的配置文件
import logging
from pathlib import Path

import yaml

from paths import CONFIGS_DIR, RUNS_DIR
from configs import COMMENTED_TRAIN_CONFIG, DEFAULT_TRAIN_CONFIG
from configs import COMMENTED_VAL_CONFIG, DEFAULT_VAL_CONFIG
from configs import COMMENTED_INFER_CONFIG, DEFAULT_INFER_CONFIG

import argparse
VALID_YOLO_TRAIN_ARGS = set(DEFAULT_TRAIN_CONFIG) # 确保只包含官方参数
VALID_YOLO_VAL_ARGS = set(DEFAULT_VAL_CONFIG)
VALID_YOLO_INFER_ARGS = set(DEFAULT_INFER_CONFIG)

BOOLEAN_PARAMS = {
    key for config in (DEFAULT_TRAIN_CONFIG, DEFAULT_VAL_CONFIG, DEFAULT_INFER_CONFIG)
        for key, value in config.items() if isinstance(value, bool)
}


logger = logging.getLogger(__name__)

def load_config(config_type= 'train'):
    """
    加载配置文件，如果文件不存在，尝试调用生成默认的配置文件，然后加载并返回
    :param config_type: 配置文件类型
    :return: 配置文件内容
    """
    config_path = CONFIGS_DIR / f'{config_type}.yaml'

    if not config_path.exists():
        logger.warning(f"配置文件{config_path}不存在，尝试生成默认的配置文件")
        if config_type in ['train', 'val', 'infer']:
            try:
                config_path.parent.mkdir(parents=True, exist_ok=True)
                generate_default_config(config_type)
                logger.info(f"生成默认的配置文件成功: {config_path}")
            except Exception as e:
                logger.error(f"创建配置文件目录失败: {e}")
                raise FileNotFoundError(f"创建配置文件目录失败: {e}")
        else:
            logger.error(f"配置文件类型错误: {config_type}")
            raise ValueError(f"配置文件类型错误: {config_type},目前仅支持train, val, infer这三种模式")

    # 加载配置文件
    try:
        logger.info(f"正在加载配置文件: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            logger.info(f"已加载配置文件: {config_path}")
            return config
    except yaml.YAMLError as e:
        logger.error(f"解析配置文件{config_path}失败: {e}")
        raise
    except Exception as e:
        logger.error(f"加载配置文件{config_path}失败: {e}")
        raise

def generate_default_config(config_type):
    """
    生成默认的配置文件
    :param config_type: 配置文件类型
    :return:
    """
    config_path = CONFIGS_DIR / f'{config_type}.yaml'
    if config_type == 'train':
        config = COMMENTED_TRAIN_CONFIG
    elif config_type == 'val':
        config = COMMENTED_VAL_CONFIG
    elif config_type == 'infer':
        config = COMMENTED_INFER_CONFIG
    else:
        logger.error(f"未知的配置文件类型：{config_type}")
        raise ValueError(f"配置文件类型错误: {config_type},目前仅支持train, val, infer这三种模式")
    try:
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write(config)
        logger.info(f"生成默认 {config_type} 配置文件成功，文件路径为：{config_path}")
    except IOError as e:
        logger.error(f"写入默认{config_type} 配置文件失败，请检查文件权限和路径是否正确，失败： {e}")
    except Exception as e:
        logger.error(f"生成配置文件 '{config_path.name}' 发生未知错误: {e}")
        raise

def _process_params_value(key, value):
    """
    辅助函数，处理常见的参数类型转换，
    例如： 'true', 'false', 转为对应的 True, False,
    :param key:
    :param value:
    :return:
    """
    # 例如： 'true', 'false', 转为对应的 True, False,
    if key in BOOLEAN_PARAMS and isinstance(value, str):
        return value.lower() == "true"
    # None, none 转为python的 None
    elif isinstance(value, str) and value.lower() == 'none':
        return None
    # 检查是否为classes参数，且内容为字符串
    elif key == "classes" and isinstance(value, str):
        if not value:
            return None
        try:
            return [int(i.strip()) for i in value.split(",")]
        except ValueError:
            logger.warning(f"参数 {key} 的值 {value} 格式不正确，请检查")
            return value
    return value

def merge_config(args,yaml_config, mode='train'):
    """
    合并命令行参数,YAML配置文件参数和默认参数,按优先级CIL > YAML > 默认值
    :param args: 通过argparse解析的参数
    :param yaml_config: 从YAML配置文件中加载的参数
    :param mode: 运行模式.支持train,val, infer
    :return:
    """
    # 1. 确定运行模式和相关配置,根据传入的mode,选择有效的YOLO参数合集
    if mode == 'train':
        valid_args = VALID_YOLO_TRAIN_ARGS
        default_config = DEFAULT_TRAIN_CONFIG
    elif mode == 'val':
        valid_args = VALID_YOLO_VAL_ARGS
        default_config = DEFAULT_VAL_CONFIG
    elif mode == 'infer':
        valid_args = VALID_YOLO_INFER_ARGS
        default_config = DEFAULT_INFER_CONFIG
    else:
        logger.error(f"{mode} 模式不存在,仅仅支持train/val/infer三种模式")
        raise ValueError(f"{mode} 模式不存在,仅仅支持train/val/infer三种模式")

    # 2. 初始化参数存储,project_args用于存储所有最终合并的参数,yolo_args用于存储yolo的参数
    project_args = argparse.Namespace()
    yolo_args = argparse.Namespace()
    merged_params = default_config.copy()

    # 3. 合并YAML参数,按优先级合并,只有当命令行指定了使用YAML文件,才进行合并,
    # 且yaml_config不是空的时候,才合并
    if hasattr(args, 'use_yaml') and args.use_yaml and yaml_config:
        for key, value in yaml_config.items():
            merged_params[key] = _process_params_value(key, value)
        logger.debug(f"合并YAML参数后: {merged_params}")

    # 4.合并命令行参数,具有最高的优先级,会覆盖YAML参数和默认值
    cmd_args = {k:v for k, v in vars(args).items() if k != 'extra_args' and v is not None}
    for key,value in cmd_args.items():
        # 未参数标记来源
        merged_params[key] = _process_params_value(key, value)
        setattr(project_args, f"{key}_specified", True)

    # # 处理动态参数
    # if hasattr(args, 'extra_args'):
    #     if len(args.extra_args) %2 != 0:
    #         logger.error("额外参数格式错误, 参数列表必须成对出现,如 --key value")
    #         raise ValueError("额外参数格式错误")
    # for i in range(0, len(args.extra_args), 2):
    #     key = args.extra_args[i].lstrip("--")
    #     value = args.extra_args[i+1]
    #
    #     processed_value = _process_params_value(key, value)
    #
    #     if processed_value == value:
    #         try:
    #             if value.replace(".", "",1).isdigit():
    #                 value = float(value) if '.' in value else int(value)
    #             elif value.lower() in ("true", "false"):
    #                 value = value.lower() == "true"
    #             elif value.lower() == "none":
    #                 value = None
    #         except ValueError:
    #             logger.warning(f"无法转换额外参数{key} 的值 {value}")
    #         merged_params[key] = value
    #     else:
    #         merged_params[key] = processed_value
    #     # 标记额外的参数来源
    #     setattr(project_args, f"{key}_specified", True)

    # 路径标准化略
    if 'data' in merged_params and merged_params['data']:
        data_path = Path(merged_params['data'])
        if not data_path.is_absolute():
            data_path = CONFIGS_DIR / data_path
        merged_params['data'] = str(data_path.resolve())
        # 验证数据集配置文件是否存在
        if not data_path.exists():
            logger.warning(f"数据集配置文件 '{data_path}' 不存在")
        logger.info(f"标准化数据集路径: '{merged_params['data']}'")

    # 标准化project参数
    if 'project' in merged_params and merged_params['project']:
        project_path = Path(merged_params['project'])
        if not project_path.is_absolute():
            project_path = RUNS_DIR / project_path
        merged_params['project'] = str(project_path)
        try:
            project_path.mkdir(parents=True, exist_ok=True)
        except PermissionError:
            logger.error(f"PermissionError: 无权限创建目录 {project_path}")
            raise ValueError(f"PermissionError: 无权限创建目录 {project_path}")
        logger.info(f"标准化project路径, {merged_params['project']}")

    # 6. 分离yolo_args和 project_args
    for key, value in merged_params.items():
        setattr(project_args, key, value)
        if key in valid_args:
            setattr(yolo_args, key, value)

        if key in yaml_config and not hasattr(project_args, f"{key}_specified"):
            setattr(project_args, f"{key}_specified", False)

    # 7.参数验证,先pass

    # 返回分离课后的两组参数
    return yolo_args, project_args




if __name__ == '__main__':
    load_config(config_type="train")