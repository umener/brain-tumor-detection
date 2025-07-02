#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :result_utils.py
# @Time      :2025/6/26 14:30:50
# @Author    :雨霓同学
# @Project   :BTD
# @Function  :
import logging
from datetime import datetime
import json
import numpy as np


# 假设 results 是 ultralytics.utils.metrics.DetMetrics 或 SegmentMetrics 对象

def log_results(results, logger=None, model_trainer=None) -> dict:  # 增加 model_trainer 参数
    """
    智能地记录 YOLO 模型（检测或分割任务）的评估结果信息。
    根据 tasks 类型自动提取并展示相关指标。

    Args:
        results: Ultralytics 的 DetMetrics 或 SegmentMetrics 对象，
                 包含模型的评估结果。
        logger: 日志记录器实例。如果为 None，将使用名为 "YOLO_Training" 的默认Logger。
        model_trainer: (可选) Ultralytics 的 Trainer 对象，用于获取 save_dir，
                       以防 results 对象中缺失或不准确。

    Returns:
        dict: 包含模型评估结果的结构化字典。
    """
    if logger is None:
        logger = logging.getLogger("YOLO_Training")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(levelname)s: %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            logger.propagate = False

    def safe_float_conversion(value, default_val=np.nan):
        if value is None:
            return default_val
        try:
            return float(value)
        except (TypeError, ValueError):
            return default_val

    # --- 1. 提取通用信息 ---
    task = getattr(results, 'task', 'unknown_task')

    # **此处是修改点：优先从 results.save_dir 获取，如果为空则尝试从 model_trainer 获取**
    save_dir_from_results = getattr(results, 'save_dir', None)
    if save_dir_from_results:
        save_dir = str(save_dir_from_results)
    elif model_trainer and hasattr(model_trainer, 'save_dir'):
        save_dir = str(model_trainer.save_dir)
        logger.info(f"从 model.trainer.save_dir 获取到模型保存路径: {save_dir}")
    else:
        save_dir = 'N/A'
        logger.warning("未能从 results 或 model.trainer 获取到有效的模型保存路径。")

    fitness = safe_float_conversion(getattr(results, 'fitness', np.nan))
    names = getattr(results, 'names', {})
    maps = getattr(results, 'maps', np.array([]))

    speed = getattr(results, 'speed', {})
    preprocess_ms = safe_float_conversion(speed.get('preprocess'))
    inference_ms = safe_float_conversion(speed.get('inference'))
    loss_ms = safe_float_conversion(speed.get('loss'))
    postprocess_ms = safe_float_conversion(speed.get('postprocess'))

    if all(not np.isnan(v) for v in [preprocess_ms, inference_ms, loss_ms, postprocess_ms]):
        total_time_ms = preprocess_ms + inference_ms + loss_ms + postprocess_ms
    else:
        total_time_ms = np.nan

    metrics_dict = getattr(results, 'results_dict', {})

    result_data = {
        "task": task,
        "save_dir": save_dir,  # 使用确定的 save_dir
        "timestamp": datetime.now().isoformat(),
        "speed_ms_per_image": {
            "preprocess": preprocess_ms,
            "inference": inference_ms,
            "loss": loss_ms,
            "postprocess": postprocess_ms,
            "total_processing": total_time_ms
        },
        "overall_metrics": {
            "fitness": fitness
        },
        "class_mAP50-95": {}
    }

    for key, value in metrics_dict.items():
        result_data["overall_metrics"][key] = safe_float_conversion(value)

    if names and maps.size > 0:
        for idx, class_name in names.items():
            if idx < maps.size:
                result_data["class_mAP50-95"][class_name] = safe_float_conversion(maps[idx])
            else:
                logger.warning(f"类别 '{class_name}' ({idx}) 没有对应的mAP值。可能 maps 数组长度不足。")
    else:
        logger.info("未获取到类别名称或类别 mAP 数据。")

    # --- 3. 日志输出 (保持与之前一致，因为它已经很完善) ---
    logger.info('=' * 60)
    logger.info(f"YOLO Results Summary ({task.capitalize()} Task)")
    logger.info('=' * 60)

    logger.info(f"{'Task':<20}: {task}")
    logger.info(f"{'Save Directory':<20}: {save_dir}")
    logger.info(f"{'Timestamp':<20}: {result_data['timestamp']}")
    logger.info('-' * 40)

    logger.info("Processing Speed (ms/image)")
    logger.info('-' * 40)
    logger.info(f"{'Preprocess':<20}: {result_data['speed_ms_per_image'].get('preprocess', np.nan):.3f} ms")
    logger.info(f"{'Inference':<20}: {result_data['speed_ms_per_image'].get('inference', np.nan):.3f} ms")
    logger.info(f"{'Loss Calc':<20}: {result_data['speed_ms_per_image'].get('loss', np.nan):.3f} ms")
    logger.info(f"{'Postprocess':<20}: {result_data['speed_ms_per_image'].get('postprocess', np.nan):.3f} ms")
    logger.info(f"{'Total Per Image':<20}: {result_data['speed_ms_per_image'].get('total_processing', np.nan):.3f} ms")
    logger.info('-' * 40)

    logger.info('Overall Evaluation Metrics')
    logger.info('-' * 40)
    logger.info(f"{'Fitness Score':<20}: {result_data['overall_metrics'].get('fitness', np.nan):.4f}")

    if task == 'detect' or task == 'segment':
        logger.info(f"{'Precision(B)':<20}: {result_data['overall_metrics'].get('metrics/precision(B)', np.nan):.4f}")
        logger.info(f"{'Recall(B)':<20}: {result_data['overall_metrics'].get('metrics/recall(B)', np.nan):.4f}")
        logger.info(f"{'mAP50(B)':<20}: {result_data['overall_metrics'].get('metrics/mAP50(B)', np.nan):.4f}")
        logger.info(f"{'mAP50-95(B)':<20}: {result_data['overall_metrics'].get('metrics/mAP50-95(B)', np.nan):.4f}")

        if task == 'segment':
            logger.info("--- Mask Metrics ---")
            logger.info(
                f"{'Precision(M)':<20}: {result_data['overall_metrics'].get('metrics/precision(M)', np.nan):.4f}")
            logger.info(f"{'Recall(M)':<20}: {result_data['overall_metrics'].get('metrics/recall(M)', np.nan):.4f}")
            logger.info(f"{'mAP50(M)':<20}: {result_data['overall_metrics'].get('metrics/mAP50(M)', np.nan):.4f}")
            logger.info(f"{'mAP50-95(M)':<20}: {result_data['overall_metrics'].get('metrics/mAP50-95(M)', np.nan):.4f}")
    else:
        logger.info(f"当前任务类型 '{task}' 的详细评估指标未完全支持。")
        for key, value in result_data["overall_metrics"].items():
            if key not in ['fitness', 'metrics/precision(B)', 'metrics/recall(B)', 'metrics/mAP50(B)',
                        'metrics/mAP50-95(B)',
                        'metrics/precision(M)', 'metrics/recall(M)', 'metrics/mAP50(M)', 'metrics/mAP50-95(M)']:
                logger.info(f"{key:<20}: {value:.4f}")

    logger.info('-' * 40)

    if result_data['class_mAP50-95']:
        logger.info("Class-wise mAP@0.5:0.95 (Box Metrics)")
        logger.info('-' * 40)
        valid_class_maps = {k: v for k, v in result_data['class_mAP50-95'].items() if not np.isnan(v)}
        if valid_class_maps:
            sorted_class_maps = sorted(valid_class_maps.items(), key=lambda item: item[1], reverse=True)
            for class_name, mAP_value in sorted_class_maps:
                logger.info(f"{class_name:<20}: {mAP_value:.4f}")
        else:
            logger.warning("所有类别 mAP 值均为 NaN，无法进行排序和打印。")
    else:
        logger.warning("未获取到类别级别的 mAP 数据。")
    logger.info('=' * 60)

    return result_data


# --- 示例用法 (与之前保持一致，并新增模拟 model_trainer) ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


    class MockTrainer:
        def __init__(self, save_dir):
            self.save_dir = save_dir


    class MockDetMetrics:
        def __init__(self, use_none_for_save_dir=False):
            self.ap_class_index = np.array([0, 1, 2])
            self.fitness = np.float64(0.32892779095815666)
            self.maps = np.array([0.31131, 0.35184, 0.26699])
            self.names = {0: 'glioma_tumor', 1: 'meningioma_tumor', 2: 'pituitary_tumor'}
            self.results_dict = {
                'metrics/precision(B)': np.float64(0.7081976737523137),
                'metrics/recall(B)': np.float64(0.15791069143002587),
                'metrics/mAP50(B)': np.float64(0.4988374355141157),
                'metrics/mAP50-95(B)': np.float64(0.3100489415630501),
                'fitness': np.float64(0.32892779095815666)
            }
            self.speed = {
                'preprocess': 0.18323289473779655,
                'inference': 2.5247460526280086,
                'loss': 0.0005986842097627232,
                'postprocess': 2.7326578947335483
            }
            self.save_dir = None if use_none_for_save_dir else 'runs/detect/train_det'
            self.task = 'detect'


    # --- 模拟正常情况 ---
    print("\n--- 检测任务结果日志 (正常) ---")
    det_results_normal = MockDetMetrics()
    logged_det_info_normal = log_results(det_results_normal)
    print("\n--- 检测任务结果 (JSON) ---")
    print(json.dumps(logged_det_info_normal, indent=4, ensure_ascii=False))

    # --- 模拟 results.save_dir 为 None，但传入 trainer 的情况 ---
    print("\n\n--- 检测任务结果日志 (results.save_dir 为 None，通过 trainer 补充) ---")
    mock_trainer = MockTrainer(save_dir='runs/detect/train_det_from_trainer')
    det_results_no_save_dir = MockDetMetrics(use_none_for_save_dir=True)
    logged_det_info_from_trainer = log_results(det_results_no_save_dir, model_trainer=mock_trainer)
    print("\n--- 检测任务结果 (JSON) ---")
    print(json.dumps(logged_det_info_from_trainer, indent=4, ensure_ascii=False))

    # --- 模拟 results.save_dir 为 None，且没有 trainer 的情况 ---
    print("\n\n--- 检测任务结果日志 (results.save_dir 为 None，且无 trainer 补充) ---")
    det_results_completely_missing_save_dir = MockDetMetrics(use_none_for_save_dir=True)
    logged_det_info_missing_all = log_results(det_results_completely_missing_save_dir)
    print("\n--- 检测任务结果 (JSON) ---")
    print(json.dumps(logged_det_info_missing_all, indent=4, ensure_ascii=False))
