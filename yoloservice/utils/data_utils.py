#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :data_utils.py
# @Time      :2025/6/24 15:56:45
# @Author    :雨霓同学
# @Project   :BTD
# @Function  :中层转换器,根据顶层用户参数,决定调用那个底层转换器
import logging
from pathlib import Path

from paths import YOLO_STAGED_LABELS_DIR
from data_converters.coco import convert_coco_json_to_yolo
from data_converters.pascal_voc import convert_pascal_voc_to_yolo

logger = logging.getLogger(__name__)

def convert_annotations_to_yolo(input_dir: Path,
                        annotation_format: str = 'coco',
                        final_classes_order = None,
                        coco_task = 'detection',
                        coco_cls91to80 = False,
                            ):
    """
    统一的标注转换入口函数,根据指定的原始标注格式,调用相应的转换器
    :param input_dir: 原始标注文件路径
    :param annotation_format: 标注文件格式,coco, pascal_voc
    :param final_classes_order: 用户指定传入的classes列表
    :param coco_task: 仅当annotation_format为coco时,coco转换任务的类型
    :param coco_cls91to80: 是否将coco 91类映射 80类
    :return: data.yaml中的names列表
    """
    logger.info(f"开始处理原始标注数据: ({annotation_format.upper()})格式 路径为: {input_dir}")

    if not input_dir.exists():
        logger.error(f"输入标注目录: {input_dir} 不存在")
        raise FileNotFoundError(f"输入目录: {input_dir} 不存在")
    classes = []
    try:
        if annotation_format == "coco":
            if final_classes_order is not None:
                logger.warning(f"COCO格式的标注数据不支持手动指定类别,目前仅支持自动提取类别")
            classes = convert_coco_json_to_yolo(
                json_input_dir=input_dir,
                task=coco_task,
                cls91to80=coco_cls91to80
            )
        elif annotation_format == "pascal_voc":
            logger.info(f"开始转换Pascal VOC格式的标注数据")
            classes = convert_pascal_voc_to_yolo(
                xml_input_dir=input_dir,
                output_yolo_txt_dir=YOLO_STAGED_LABELS_DIR,
                target_classes_for_yolo=final_classes_order
            )
            if not classes:
                logger.error(f"转换Pascal VOC格式的标注数据时失败,为提取到任何类别")
                return []
            logger.info(f"转换完成，转换的类别为：{classes}")
        else:
            logger.error(f"不支持的标注格式: {annotation_format},目前仅支持: 'coco' 或 'pascal_voc'")
            raise ValueError(f"不支持的标注格式: {annotation_format},目前仅支持: 'coco' 或 'pascal_voc'")
    except Exception as e:
        logger.critical(f"转换Pascal VOC格式的标注数据发生致命错误,"
                        f"格式{annotation_format},错误信息为: {e}",exc_info=True)
        classes = []
    if not classes:
        logger.warning(f"数据转换完成,但是未能确定任何可用的类别,请检查你的数据")
    logger.info(f"标注格式{annotation_format.upper()}转换处理完成")
    return classes





