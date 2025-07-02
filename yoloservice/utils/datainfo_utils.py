#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :datainfo_utils.py
# @Time      :2025/6/26 14:33:25
# @Author    :雨霓同学
# @Project   :BTD
# @Function  :
#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :dataset_utils.py
# @Time      :2025/6/12 10:14:48
# @Author    :雨霓同学
# @Project   :SafeH
# @Function  :用于获取数据集信息
import json

import yaml
import logging
from pathlib import Path
from utils import CONFIGS_DIR  # 假设 CONFIGS_DIR 在这里被正确导入并保证其正确性

# 常见图片文件扩展名列表，可以根据需要扩展
COMMON_IMAGE_EXTENSIONS = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff', '*.bmp', '*.webp']


def get_dataset_info(data_config_name: str, mode: str = "train") -> tuple[int, list[str], int, str]:
    """
    获取数据集信息，包括类别数，类别名称和样本数量。
    能够处理不同模式下（train, val, test, infer）的数据集路径和样本统计。
    假定 data_config_name 文件位于 CONFIGS_DIR 下。

    Args:
        data_config_name (str): 数据集的配置文件名称（如 "data.yaml"）。
        mode (str): 模式，可选值为 "train", "val", "test", "infer"。

    Returns:
        tuple[int, list[str], int, str]: (类别数, 类别名称列表, 样本数, 样本来源描述)。
    """
    logger = logging.getLogger("YOLO_Training")

    # 初始化返回值
    nc: int = 0
    classes_names: list[str] = []
    samples: int = 0
    source: str = "未知"

    # 推理模式下不提供数据集来源信息，直接返回
    if mode == 'infer':
        return 0, [], 0, "推理模式，不提供数据集来源信息"

    # 获取数据集配置文件路径，基于全局 CONFIGS_DIR
    data_path: Path = CONFIGS_DIR / data_config_name

    # 读取数据集配置文件
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"数据集配置文件 '{data_path}' 不存在。请检查 CONFIGS_DIR 或文件名称是否正确。")
        return nc, classes_names, samples, f"配置文件不存在: {data_path}"
    except yaml.YAMLError as e:
        logger.error(f"读取或解析数据集配置文件 '{data_path}' 失败: {e}")
        return nc, classes_names, samples, f"配置文件解析失败: {data_path}"
    except Exception as e:
        logger.error(f"打开或读取数据集配置文件 '{data_path}' 时发生未知错误: {e}")
        return nc, classes_names, samples, f"配置文件读取错误: {data_path}"

    # 获取类别数和类别名称
    nc = config.get("nc", 0)
    classes_names = config.get("names", [])

    # 根据模式确定图片路径
    split_key: str = mode
    # YOLO通常会将数据集路径定义为相对于config文件
    dataset_root_from_config = config.get("path")  # 获取数据yaml中定义的root路径

    # 获取具体split的相对路径
    split_relative_path_str: str = config.get(split_key)

    # 如果配置文件中未定义特定模式的路径，则使用默认约定
    if not split_relative_path_str:
        logger.warning(
            f"配置文件 '{data_config_name}' 中未定义 '{split_key}' 模式的图片路径。尝试使用默认约定 '{mode}/images'。")
        split_relative_path_str = f"{mode}/images"

    # 构建完整的图片目录路径
    # 优先使用配置文件中定义的 'path' 作为数据集根目录
    if dataset_root_from_config:
        dataset_base_path = Path(dataset_root_from_config)
        # 如果 path 是相对路径，则相对于配置文件所在目录
        if not dataset_base_path.is_absolute():
            dataset_base_path = data_path.parent / dataset_root_from_config
    else:
        # 如果没有定义 'path'，则假定图片目录与配置文件在同一层或其子目录
        dataset_base_path = data_path.parent
        # 移除冗余日志，因为这在某些情况下是正常约定
        # logger.info(f"配置文件 '{data_config_name}' 中未定义 'path'。假定数据集根目录为 '{dataset_base_path}'。")

    split_path: Path = dataset_base_path / split_relative_path_str

    # 检查图片路径是否存在
    if split_path.is_dir():
        # 统计样本数量
        for ext in COMMON_IMAGE_EXTENSIONS:
            samples += len(list(split_path.glob(ext)))
        source = f"{mode.capitalize()} images from: {split_path}"
        # 移除对 samples == 0 的警告，因为这可能是预期的（比如某个 split 没数据）
    else:
        logger.error(f"数据集图片路径不存在或不是一个目录: '{split_path}'。请检查配置文件中的路径或数据集完整性。")
        source = f"{mode.capitalize()} images not found at: {split_path}"

    return nc, classes_names, samples, source


def log_dataset_info(data_config_name: str, mode: str = 'train', logger: logging.Logger = None) -> dict:
    """
    获取并记录数据集信息到日志。

    Args:
        data_config_name (str): 数据集的配置文件名称。
        mode (str): 模式，可选值为 "train", "val", "test", "infer"。
        logger (logging.Logger, optional): 用于记录信息的日志记录器实例。
                                          默认为None，此时会获取名为"YOLO_Training"的logger。

    Returns:
        dict: 结构化的数据集信息字典。
    """
    if logger is None:
        logger = logging.getLogger("YOLO_Training")
        # 确保日志记录器已配置处理器且不会重复传播
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)s: %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            logger.propagate = False  # 阻止传播到根记录器

    nc, classes_names, samples, source = get_dataset_info(data_config_name, mode)

    logger.info("=".center(40, '='))
    logger.info(f"数据集信息 ({mode.capitalize()} 模式)")
    logger.info('-' * 40)
    logger.info(f"{'Config File':<20}: {data_config_name}")
    logger.info(f"{'Class Count':<20}: {nc}")
    logger.info(f"{'Class Names':<20}: {', '.join(classes_names) if classes_names else '未知'}")
    logger.info(f"{'Sample Count':<20}: {samples}")
    logger.info(f"{'Data Source':<20}: {source}")
    logger.info('-' * 40)

    return {
        "config_file": data_config_name,
        "mode": mode,  # 记录模式，方便追溯
        "class_count": nc,
        "class_names": classes_names,
        "sample_count": samples,
        "data_source": source
    }


if __name__ == "__main__":
    # 配置logging，确保在主程序运行时日志能正确输出
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # 示例用法
    # 注意：运行此脚本需要确保你的项目结构中
    # 1. 存在 `your_project_root/utils/__init__.py` 或其他地方定义了 `CONFIGS_DIR`
    # 2. `CONFIGS_DIR` 指向正确的 `configs` 目录
    # 3. `configs` 目录下有 `data.yaml` 文件，且内容符合 YOLO 数据集配置格式。

    # 例如，如果你的 data.yaml 位于 `项目根目录/configs/data.yaml`，且内容如下：

    print("--- 获取训练集信息 ---")
    dataset_train_info = log_dataset_info(data_config_name="data.yaml", mode="train")
    print("--- 获取验证集信息 ---")
    dataset_val_info = log_dataset_info(data_config_name="data.yaml", mode="val")
    print("--- 获取测试集信息 ---")
    dataset_test_info = log_dataset_info(data_config_name="data.yaml", mode="test")
    print("--- 获取推理模式信息 ---")
    dataset_infer_info = log_dataset_info(data_config_name="data.yaml", mode="infer")

    print("--- 训练集信息 (JSON) ---")
    print(json.dumps(dataset_train_info, indent=4, ensure_ascii=False))
