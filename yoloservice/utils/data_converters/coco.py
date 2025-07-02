#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :coco.py
# @Time      :2025/6/24 14:40:16
# @Author    :雨霓同学
# @Project   :BTD
# @Function  : 主要实现coco json转yolo格式，并返回一个名称列表，
import json
import logging
import datetime
import shutil
from pathlib import Path

from ultralytics.data.converter import convert_coco

from paths import RAW_DATA_DIR, YOLO_STAGED_LABELS_DIR

logger = logging.getLogger(__name__)

def convert_coco_json_to_yolo(json_input_dir: Path, task: str = "detection",cls91to80: bool = False):
    """
    将coco json标注文件转换为yolo格式的txt文件到指定的目录
    :param cls91to80: 是否将coco 91类映射到80类
    :param json_input_dir: 包含coco json的目录
    :param task: 任务类型
    :return: class_name:检测对象名称列表
    """
    logger.info(f"开始转换coco json标注文件,从{json_input_dir} 转为YOLO格式【自动模式】")

    if not json_input_dir.exists():
        logger.error(f"coco json输入目录: {json_input_dir} 不存在")
        raise FileNotFoundError(f"coco json输入目录: {json_input_dir} 不存在")

    # 1. 查找目录中所有的Json文件并提示数量
    json_files_found = list(json_input_dir.glob("*.json"))
    if not json_files_found:
        logger.error(f"coco json输入目录: {json_input_dir} 中不存在json文件")
        raise FileNotFoundError(f"coco json输入目录: {json_input_dir} 中不存在json文件")
    logger.info(f"coco json输入目录: {json_input_dir} 中找到 {len(json_files_found)} 个json文件")

    # 2. 判断json文件的 'categories' 是否相同并收集使用 category_id
    first_categories_set = set()
    first_coco_json_path = json_files_found[0]
    all_used_category_ids = set()
    original_coco_id_to_name_map = {}

    for i, json_file_path in enumerate(json_files_found):
        try:
            with open(json_file_path, "r", encoding="utf-8") as f:
                current_coco_data = json.load(f)
            current_categories_set = set()
            for cat in current_coco_data.get('categories', []):
                if 'id' in cat and 'name' in cat:
                    current_categories_set.add((cat['id'], cat['name']))
            for ann in current_coco_data.get('annotations', []):
                if 'category_id' in ann:
                    all_used_category_ids.add(ann['category_id'])

            if i == 0:
                first_categories_set = current_categories_set
                for cat in current_coco_data.get('categories', []):
                    if 'id' in cat and 'name' in cat:
                        original_coco_id_to_name_map[cat['id']] = cat['name']
                logger.info(f"已加载基准json文件：'{json_file_path.name}'的categories信息，"
                            f"并构建原始ID到名称的映射关系。")
            else:
                if first_categories_set != current_categories_set:
                    logger.critical(f"数据集存在严重错误！Json文件 '{json_file_path.name}' "
                                f"的categories信息与第一个'{first_coco_json_path.name}'文件不一致！请检查！")
                    raise ValueError(f"数据集存在严重错误！Json文件 '{json_file_path.name}' "
                                f"的categories信息与第一个'{first_coco_json_path.name}'文件不一致！请检查！")
                logger.info(f"Json文件 '{json_file_path.name}'的categories 与定义的基准文件一致！")
        except json.JSONDecodeError as e:
            logger.error(f"Json文件 '{json_file_path.name}' 解析错误！: {e}")
            raise
        except Exception as e:
            logger.error(f"读取或者处理 coco json文件 {json_file_path.name} 时发生错误！: {e}")
            raise
    # 3. 提取实际使用的类别ID并构建最终的classes_name列表，用于data.yaml的names字段
    sorted_used_categories = sorted(list(all_used_category_ids))

    classes_name = []
    for cat_id in sorted_used_categories:
        if cat_id in original_coco_id_to_name_map:
            classes_name.append(original_coco_id_to_name_map[cat_id])
        else:
            logger.warning(f"在 annotations 中发现 category_id {cat_id} "
                        f"但是在 categories 中没有找到对应的名称，将跳过此ID")
    if not classes_name:
        logger.error("未能从所有的json文件中的annotations中找到任何类别，转换终止")
        return []
    logger.info(f"根据所有的JSon文件的annotations构建的最终列表为：{classes_name}")
    if cls91to80:
        logger.info(f"注意：'cls91to80'参数为True,ultralytics将在内部对列表ID进行映射，"
                    f"但是本函数返回到classes_name列表是基于原始coco列表和标注中使用情况决定的")

    # 4. 定义文件处理逻辑
    # 1. 生成一个基于当前时间唯一的临时目录名
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S_%f")
    temp_dir_name = f"{timestamp}"
    temp_dir = RAW_DATA_DIR / temp_dir_name

    use_segments = (task == "segmentation")

    # 2. 调用ultralytics的coco2yolo函数
    try:
        _ = convert_coco(
            labels_dir=str(json_input_dir),
            save_dir = str(temp_dir),
            use_segments=use_segments,
            use_keypoints=False,
            cls91to80=cls91to80,
            lvis=False
        )
        logger.info(f"ultralytics.convert_coco转换完成到临时目录{temp_dir}")
    except Exception as e:
        logger.critical(f"转换失败，请检查数据集格式是否正确，错误信息为：{e}")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        return []
    # 3. 剪切coco转换的数据到 指定的临时存放点
    source_labels_in_temp = temp_dir / "labels"
    if not source_labels_in_temp.exists():
        logger.error(f"临时转换目录{temp_dir} 中不存在 labels目录，可能是因为转换失败原因")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
            return []
    # 确保YOLO_STAGED_LABELS_DIR 存在
    YOLO_STAGED_LABELS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"开始将 生成的TXT 文件从{source_labels_in_temp} 剪切到 {YOLO_STAGED_LABELS_DIR} 中...")

    moved_count = 0
    for txt_file in source_labels_in_temp.glob("./*/*.txt"):
        try:
            shutil.move(str(txt_file), str(YOLO_STAGED_LABELS_DIR / txt_file.name))
            moved_count += 1
        except Exception as e:
            logger.error(f"移动文件{txt_file.name} 到 {YOLO_STAGED_LABELS_DIR}失败，错误信息为：{e}")
        logger.info(f"成功移动了{moved_count}个YOLO TXT 文件，到 {YOLO_STAGED_LABELS_DIR}")

    # 删除临时目录
    try:
        shutil.rmtree(temp_dir)
        logger.info(f"成功删除临时目录 {temp_dir}")
    except Exception as e:
        logger.error(f"删除临时目录{source_labels_in_temp}失败，错误信息为：{e}")
    logger.info(f"COCO JSON 到 YOLO TXT 转换流程完成".center(60, "="))

    return classes_name


if __name__=="__main__":
    classes_name_ = convert_coco_json_to_yolo(
        json_input_dir=Path(r"D:\BTD\yoloservice\data\raw\original_annotations"),
        task = "detection",
    )
    print(classes_name_)
