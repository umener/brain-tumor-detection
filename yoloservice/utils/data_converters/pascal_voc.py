#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :pascal_voc.py
# @Time      :2025/6/22 09:28:41
# @Author    :雨霓同学
# @Project   :DTH2
# @Function  :

import xml.etree.ElementTree as ET
import logging
from pathlib import Path
from typing import List, Dict, Any, Union, Set # 引入 Set 类型提示


# 配置日志 (假设在模块外部或主程序入口已完成，这里只获取)
logger = logging.getLogger(__name__)

# --- 辅助函数：解析 Pascal VOC XML 到 YOLO 格式 (保持不变，因为主要关注COCO) ---
def _parse_xml_annotation(xml_path: Path, classes: List[str]) -> List[str]:
    # ... 此函数保持不变 ...
    yolo_labels = []
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size_elem = root.find("size")

        if size_elem is None:
            logger.error(f"XML文件 '{xml_path.name}' 缺少 'size' 元素，无法提取图像尺寸信息，跳过。")
            return []

        width = int(size_elem.find("width").text)
        height = int(size_elem.find("height").text)

        if width <= 0 or height <= 0:
            logger.warning(f"XML文件 '{xml_path.name}' 的图像尺寸信息无效 (W:{width}, H:{height})，跳过。")
            return []

        for obj in root.iter("object"):
            name_elem = obj.find("name")
            if name_elem is None or not name_elem.text:
                logger.warning(f"XML文件 '{xml_path.name}' 发现缺少 'name' 标签的对象，跳过。")
                continue

            name = name_elem.text.strip()

            if name not in classes:
                logger.debug(f"XML文件 '{xml_path.name}' 的类别 '{name}' 不在目标类别列表中，跳过该标注。")
                continue

            class_id = classes.index(name)
            xml_box = obj.find("bndbox")
            if xml_box is None:
                logger.warning(f"XML文件 '{xml_path.name}' 对象 '{name}' 缺少 'bndbox' 标签，跳过。")
                continue

            try:
                x_min = int(xml_box.find("xmin").text)
                y_min = int(xml_box.find("ymin").text)
                x_max = int(xml_box.find("xmax").text)
                y_max = int(xml_box.find("ymax").text)
            except (AttributeError, ValueError) as e:
                logger.warning(f"XML文件 '{xml_path.name}' 对象 '{name}' 的边界框坐标解析失败: {e}，跳过。")
                continue

            if not (0 <= x_min < x_max <= width and 0 <= y_min < y_max <= height):
                logger.warning(
                    f"XML文件 '{xml_path.name}' 的标注坐标无效 (xmin:{x_min}, ymin:{y_min}, xmax:{x_max}, ymax:{y_max}, img_w:{width}, img_h:{height})，跳过该标注框。"
                )
                continue

            center_x = (x_min + x_max) / (2.0 * width)
            center_y = (y_min + y_max) / (2.0 * height)
            box_width = (x_max - x_min) / width
            box_height = (y_max - y_min) / height

            center_x = max(0.0, min(1.0, center_x))
            center_y = max(0.0, min(1.0, center_y))
            box_width = max(0.0, min(1.0, box_width))
            box_height = max(0.0, min(1.0, box_height))

            yolo_labels.append(f"{class_id} {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}")
        return yolo_labels

    except FileNotFoundError:
        logger.error(f"XML文件 '{xml_path}' 不存在，跳过该文件。")
    except ET.ParseError as e:
        logger.error(f"XML文件 '{xml_path.name}' 解析错误，失败: {e}")
    except Exception as e:
        logger.error(f"解析XML文件 '{xml_path.name}' 发生未知错误: {e}")
    return []


# --- 核心转换函数：Pascal VOC XML 到 YOLO TXT (保持不变) ---
def convert_pascal_voc_to_yolo(xml_input_dir: Path, output_yolo_txt_dir: Path,
                            target_classes_for_yolo: Union[List[str], None] = None) -> List[str]:
    # ... 此函数保持不变 ...
    logger.info(f"== 开始将Pascal VOC XML文件从 '{xml_input_dir}' 转换为YOLO格式并保存到 '{output_yolo_txt_dir}' ==")

    if not xml_input_dir.exists():
        logger.error(f"XML输入目录不存在: {xml_input_dir}")
        raise FileNotFoundError(f"XML输入目录不存在: {xml_input_dir}")

    xml_files_found = list(xml_input_dir.glob("*.xml"))
    if not xml_files_found:
        logger.warning(f"在目录 '{xml_input_dir}' 中未找到任何XML标注文件。")
        return []

    classes: List[str] = []
    if target_classes_for_yolo is not None:
        classes = target_classes_for_yolo
        logger.info(f"Pascal VOC转换模式: 手动模式。使用指定类别: {classes}")
    else:
        unique_classes = set()
        logger.info("Pascal VOC转换模式: 自动模式。开始扫描XML文件以收集所有类别名称...")
        for xml_file in xml_files_found:
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()
                for obj in root.iter("object"):
                    name_elem = obj.find("name")
                    if name_elem is not None and name_elem.text:
                        unique_classes.add(name_elem.text.strip())
            except ET.ParseError as e:
                logger.warning(f"扫描XML文件 '{xml_file.name}' 时解析错误: {e}")
            except Exception as e:
                logger.warning(f"扫描XML文件 '{xml_file.name}' 时发生未知错误: {e}")

        classes = sorted(list(unique_classes))
        if not classes:
            logger.error(f"从XML文件 '{xml_input_dir}' 中未提取到任何类别名称，请检查XML文件内容。")
            return []
        logger.info(f"Pascal VOC转换模式: 自动模式。动态提取类别: {classes}")

    output_yolo_txt_dir.mkdir(parents=True, exist_ok=True)
    converted_count = 0

    for xml_file in xml_files_found:
        yolo_labels = _parse_xml_annotation(xml_file, classes)
        if yolo_labels:
            txt_file_path = output_yolo_txt_dir / (xml_file.stem + ".txt")
            try:
                with open(txt_file_path, "w", encoding="utf-8") as f:
                    for label in yolo_labels:
                        f.write(label + "\n")
                converted_count += 1
            except Exception as e:
                logger.error(f"写入YOLO标签文件 '{txt_file_path.name}' 失败: {e}")
        else:
            logger.debug(f"文件 '{xml_file.name}' 未生成有效的YOLO标签，可能无目标类别或解析失败。")

    logger.info(f"从 '{xml_input_dir}' 成功转换了 {converted_count} 个XML文件的标注到YOLO格式。")
    return classes
