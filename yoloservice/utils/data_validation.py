#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :dataset_validation.py
# @Time      :2025/6/14 21:30:48
# @Author    :雨霓同学
# @Project   :BrainTumorDetection
# @Function  :验证YOLO数据集配置以及相关文件

import yaml
from pathlib import Path
import logging
import random
import shutil  # 新增导入 shutil 用于文件删除

# 配置验证模式和参数
SAMPLE_SIZE = 0.1
MIN_SAMPLES = 10

# 日志临时记录
logger = logging.getLogger("YOLO_DatasetVerification")
from performance_utils import time_it  # 假设 logging_utils 和 time_it 已经正确配置


@time_it(name="数据集配置验证", logger_instance=logger)
def verify_dataset_config(yaml_path: Path, current_logger: logging.Logger, mode: str = "SAMPLE",
                        task_type: str = "detection"):
    """
    验证YOLO数据集配置，检查data.yaml和对应的图像、标签文件。
    根据 task_type 参数验证标签文件格式。

    :param yaml_path: data.yaml的路径。
    :param current_logger: 配置好的logger实例。
    :param mode: 验证模式，默认为 "SAMPLE"，可选 "FULL"。
    :param task_type: 任务类型，"detection" 或 "segmentation"。
    :return: Tuple[bool, List[Dict]]: 如果验证通过返回 (True, [])，否则返回 (False, invalid_samples_list)。
            invalid_samples_list 中的每个字典包含 'image_path', 'label_path', 'error_message'。
    """
    current_logger.info(f"验证data.yaml文件配置,配置文件路径为：{yaml_path}")
    current_logger.info(f"当前验证任务类型为: {task_type.upper()}")

    invalid_samples = []  # 用于收集不合法样本信息

    if not yaml_path.exists():
        current_logger.error(f"data.yaml文件不存在: {yaml_path}，请检查配置文件路径是否正确")
        return False, invalid_samples

    # 读取YAML文件
    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        current_logger.error(f"读取data.yaml文件失败: {e}")
        return False, invalid_samples

    classes_names = config.get("names", [])
    nc = config.get("nc", 0)

    if len(classes_names) != nc:
        current_logger.error(f"数据集类别数量与配置文件不一致，{len(classes_names)} != {nc},请检查配置文件")
        return False, invalid_samples
    current_logger.info(f"数据集类别数量与配置文件一致，类别数量为：{nc}，类别为：{classes_names}")

    # 验证数据集
    splits = ["train", "val"]
    if 'test' in config and config['test'] is not None:
        splits.append('test')
    else:
        current_logger.info("data.yaml中未定义 'test'路径或其路径值为None，跳过test验证")

    overall_validation_status = True  # 用于跟踪总体验证状态

    for split in splits:
        split_path = Path(config[split]).resolve()

        current_logger.info(f"验证 {split} 路径为: {split_path}")
        if not split_path.exists():
            current_logger.error(f"{split} 路径不存在: {split_path}")
            overall_validation_status = False
            # 对于路径不存在的情况，不记录到 invalid_samples，因为这不是具体文件的问题
            continue

        # 获取图像文件
        img_paths = (
                list(split_path.glob("*.[jJ][pP][gG]")) +
                list(split_path.glob("*.[pP][nN][gG]")) +
                list(split_path.glob("*.[jJ][pP][eE][gG]")) +
                list(split_path.glob("*.[tT][iI][fF]")) +
                list(split_path.glob("*.[tT][iI][fF][fF]")) +
                list(split_path.glob("*.[bB][mM][pP]")) +
                list(split_path.glob("*.[wW][eE][bB][pP]"))  # 添加webp支持
        )
        if not img_paths:
            current_logger.error(f"图像目录{split_path} 路径下没有图像文件")
            overall_validation_status = False
            continue
        current_logger.info(f"图像目录{split_path} 存在{len(img_paths)}张图像")

        # 动态抽样
        sample_size = max(MIN_SAMPLES, int(len(img_paths) * SAMPLE_SIZE))
        if mode == "FULL":
            current_logger.info(f"{split} 验证模式为FULL，将验证所有图像")
            sample_paths = img_paths
        else:
            current_logger.info(f"{split} 验证模式为SAMPLE，将随机抽取{sample_size}张图像进行验证")
            sample_paths = random.sample(img_paths, min(sample_size, len(img_paths)))

        # 验证图像和文件标签
        for img_path in sample_paths:
            img_path_resolve = img_path.resolve()
            current_logger.debug(f"验证图像文件: {img_path_resolve}")

            # 假设标签目录与图像目录同级，但命名为 'labels'
            # 例如：图片在 data/train/images 下，标签在 data/train/labels 下
            # 或者：图片在 data/images/train 下，标签在 data/labels/train 下
            # 这里保持原逻辑：img_path.parent 是 'images' 文件夹，img_path.parent.parent 是 'train' 文件夹
            # label_dir 则指向 'train/labels'
            label_dir = img_path.parent.parent / "labels"  # 这条路径是正确的，指向如 data/train/labels

            label_path = label_dir / (img_path.stem + ".txt")

            current_logger.debug(f"验证标签文件: {label_path}")
            if not label_path.exists():
                current_logger.error(f"标签文件不存在: {label_path}，无法为图像 {img_path.name} 找到对应的标签。")
                overall_validation_status = False
                invalid_samples.append({
                    "image_path": img_path_resolve,
                    "label_path": label_path,
                    "error_message": "标签文件不存在"
                })
                continue

            # 验证标签内容是否正确
            try:
                with open(label_path, "r", encoding="utf-8") as f:
                    lines = f.read().splitlines()
            except Exception as e:
                current_logger.error(f"读取标签文件失败: {label_path}, 错误: {e}")
                overall_validation_status = False
                invalid_samples.append({
                    "image_path": img_path_resolve,
                    "label_path": label_path,
                    "error_message": f"读取标签文件失败: {e}"
                })
                continue

            if not lines:
                current_logger.warning(f"标签文件为空: {label_path}, 表示图像 {img_path.name} 没有标注。")
                # 空标签文件不一定是不合法，取决于数据集定义。此处不加入 invalid_samples，仅警告
                continue

            # 标记当前标签文件是否有错误，避免重复添加 invalid_samples
            current_label_has_error = False
            for line_idx, line in enumerate(lines):
                parts = line.split(" ")

                is_format_correct = True
                error_detail = ""
                if task_type == "detection":
                    # 检测任务：class_id x_center y_center width height (5个值)
                    if len(parts) != 5:
                        error_detail = "不符合检测 YOLO 格式 (应为5个浮点数)"
                        is_format_correct = False
                elif task_type == "segmentation":
                    # 分割任务：class_id x1 y1 x2 y2 ... xN yN (至少7个值，即 1 + 2*N, N >= 3)
                    if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
                        error_detail = "不符合分割 YOLO 格式 (应为至少7个值，且类别ID后坐标对数量为偶数)"
                        is_format_correct = False
                else:
                    error_detail = f"未知的任务类型 '{task_type}'"
                    is_format_correct = False

                if not is_format_correct:
                    current_logger.error(
                        f"标签文件格式错误: {label_path}，行 {line_idx + 1}: '{line}' {error_detail}。"
                    )
                    overall_validation_status = False
                    current_label_has_error = True
                    invalid_samples.append({
                        "image_path": img_path_resolve,
                        "label_path": label_path,
                        "error_message": f"标签格式错误: {error_detail}, 行 '{line_idx + 1}: {line}'"
                    })
                    break  # 跳出当前标签文件的行循环，因为该文件已发现格式问题

                try:
                    class_id = int(parts[0])
                    # 检查 class_id 是否在有效范围内
                    if not (0 <= class_id < nc):
                        current_logger.error(
                            f"标签文件 {label_path} 内容错误: 类别ID {class_id} 超出 [0, {nc - 1}] 范围。行 {line_idx + 1}: '{line}'")
                        overall_validation_status = False
                        current_label_has_error = True
                        invalid_samples.append({
                            "image_path": img_path_resolve,
                            "label_path": label_path,
                            "error_message": f"类别ID超出范围: {class_id}, 行 '{line_idx + 1}: {line}'"
                        })
                        break

                    # 验证坐标值是否在 [0,1] 范围内
                    # 注意：对于分割，coords 将包含所有多边形点
                    coords = [float(x) for x in parts[1:]]
                    if not all(0 <= x <= 1 for x in coords):
                        current_logger.error(
                            f"标签文件 {label_path} 内容错误，坐标 {coords} 超出 [0,1] 范围。行 {line_idx + 1}: '{line}'")
                        overall_validation_status = False
                        current_label_has_error = True
                        invalid_samples.append({
                            "image_path": img_path_resolve,
                            "label_path": label_path,
                            "error_message": f"坐标值超出 [0,1] 范围: {coords}, 行 '{line_idx + 1}: {line}'"
                        })
                        break
                except ValueError:
                    current_logger.error(f"标签文件 {label_path} 包含无效值 (非数字): '{line}'")
                    overall_validation_status = False
                    current_label_has_error = True
                    invalid_samples.append({
                        "image_path": img_path_resolve,
                        "label_path": label_path,
                        "error_message": f"标签包含非数字值: '{line_idx + 1}: {line}'"
                    })
                    break

            # 如果当前标签文件有错误，跳过该文件的后续行检查，直接处理下一个图像文件
            if current_label_has_error:
                continue

    # 最后判断总体验证状态
    if invalid_samples:
        current_logger.error(
            f"基础数据集结构或标签内容验证未通过，共检测到 {len(invalid_samples)} 个不合法的图像-标签对。请检查日志中的错误。")
        overall_validation_status = False  # 如果有不合法样本，则总体验证状态为未通过
    else:
        current_logger.info(f"基础数据集结构和标签内容验证通过！")

    return overall_validation_status, invalid_samples


@time_it(name="数据集分割验证", logger_instance=logger)
def verify_split_uniqueness(yaml_path: Path, current_logger: logging.Logger):
    """
    验证数据集划分（train, val, test）之间是否存在重复图像。

    :param yaml_path: data.yaml 的路径。
    :param current_logger: 配置好的 logger 实例。
    :return: bool: 如果没有重复返回 True，否则返回 False。
    """
    current_logger.info("开始验证数据集划分的唯一性（train, val, test 之间无重复图像）。")
    if not yaml_path.exists():
        current_logger.error(f"data.yaml 文件不存在: {yaml_path}，无法进行分割唯一性验证。")
        return False

    try:
        with open(yaml_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        current_logger.error(f"读取data.yaml文件失败: {e}，无法进行分割唯一性验证。")
        return False

    split_image_stems = {}  # 存储每个分割的图像文件名（不含扩展名）集合

    splits = ["train", "val"]
    if 'test' in config and config['test'] is not None:
        splits.append('test')

    overall_uniqueness_status = True

    for split in splits:
        split_path_str = config.get(split)
        if split_path_str is None:
            current_logger.warning(f"data.yaml 中未定义 '{split}' 路径，跳过该分割的唯一性检查。")
            continue

        split_path = Path(split_path_str).resolve()

        if not split_path.exists():
            current_logger.error(f"'{split}' 图像路径不存在: {split_path}，无法进行唯一性验证。")
            overall_uniqueness_status = False
            continue

        # 获取所有图像文件的文件名（不含扩展名）
        img_stems = set()
        img_extensions = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.tif", "*.webp"]
        for ext in img_extensions:
            for img_file in split_path.glob(ext):
                img_stems.add(img_file.stem)

        split_image_stems[split] = img_stems
        current_logger.info(f"'{split}' 分割包含 {len(img_stems)} 张独立图像。")

    # 检查集合之间的交集
    # train 和 val 之间
    if "train" in split_image_stems and "val" in split_image_stems:
        common_train_val = split_image_stems["train"].intersection(split_image_stems["val"])
        if common_train_val:
            current_logger.error(
                f"警告：训练集和验证集之间发现重复图像！重复数量: {len(common_train_val)}。示例: {list(common_train_val)[:5]}")
            overall_uniqueness_status = False
        else:
            current_logger.info("训练集和验证集之间没有重复图像。")

    # train 和 test 之间
    if "train" in split_image_stems and "test" in split_image_stems:
        common_train_test = split_image_stems["train"].intersection(split_image_stems["test"])
        if common_train_test:
            current_logger.error(
                f"警告：训练集和测试集之间发现重复图像！重复数量: {len(common_train_test)}。示例: {list(common_train_test)[:5]}")
            overall_uniqueness_status = False
        else:
            current_logger.info("训练集和测试集之间没有重复图像。")

    # val 和 test 之间
    if "val" in split_image_stems and "test" in split_image_stems:
        common_val_test = split_image_stems["val"].intersection(split_image_stems["test"])
        if common_val_test:
            current_logger.error(
                f"警告：验证集和测试集之间发现重复图像！重复数量: {len(common_val_test)}。示例: {list(common_val_test)[:5]}")
            overall_uniqueness_status = False
        else:
            current_logger.info("验证集和测试集之间没有重复图像。")

    if overall_uniqueness_status:
        current_logger.info("数据集分割唯一性验证通过：各子集之间没有发现重复图像。")
    else:
        current_logger.error("数据集分割唯一性验证未通过，存在重复图像。请检查日志。")

    return overall_uniqueness_status


def delete_invalid_files(invalid_data_list: list, current_logger: logging.Logger):
    """
    删除列表中指定的不合法图像和标签文件。

    :param invalid_data_list: 包含不合法数据路径的列表，每个元素应包含 'image_path' 和 'label_path'。
    :param current_logger: 日志实例。
    """
    current_logger.info("开始删除不合法的图像和标签文件...")
    deleted_image_count = 0
    deleted_label_count = 0

    for item in invalid_data_list:
        img_path = item['image_path']
        label_path = item['label_path']

        # 记录是哪个错误导致的文件被标记为不合法
        error_msg = item.get('error_message', '未知错误')

        current_logger.debug(f"尝试删除因 '{error_msg}' 而被标记为不合法的图像: {img_path} 及其标签: {label_path}")

        try:
            if img_path.exists():
                img_path.unlink()  # 删除文件
                current_logger.info(f"成功删除图像文件: {img_path}")
                deleted_image_count += 1
            else:
                current_logger.warning(f"图像文件不存在，跳过删除: {img_path}")

            if label_path.exists():
                label_path.unlink()  # 删除文件
                current_logger.info(f"成功删除标签文件: {label_path}")
                deleted_label_count += 1
            else:
                current_logger.warning(f"标签文件不存在，跳过删除: {label_path}")

        except OSError as e:
            current_logger.error(f"删除文件失败: {e} - 无法删除图像: {img_path} 或标签: {label_path}")

    current_logger.info(f"删除操作完成。共删除 {deleted_image_count} 个图像文件和 {deleted_label_count} 个标签文件。")