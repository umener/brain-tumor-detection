#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :yolo_validate.py
# @Time      :2025/6/11 16:35:11
# @Author    :雨霓同学
# @Project   :SafeH
# @Function  :YOLO数据集验证脚本，调用utils模块中的验证逻辑

import logging
import sys
from pathlib import Path
import argparse  # 新增：导入argparse模块用于处理命令行参数


# 将项目根目录和utils模块路径添加到Python的搜索路径中，确保可以正确导入
current_path = Path(__file__).parent.parent.resolve()
utils_path = current_path / 'utils'
if str(current_path) not in sys.path:
    sys.path.insert(0, str(current_path))
if str(utils_path) not in sys.path:
    sys.path.insert(1, str(utils_path))

# 从utils模块中导入数据集验证逻辑和常量
from utils.data_validation import (
    verify_dataset_config,
    verify_split_uniqueness,
    delete_invalid_files,  # 导入delete_invalid_files函数
)
from utils.logging_utils import setup_logger
from utils.paths import LOGS_DIR, CONFIGS_DIR


if __name__ == "__main__":
    # 使用argparse解析命令行参数
    parser = argparse.ArgumentParser(description="YOLO数据集验证脚本。")
    parser.add_argument(
        '--mode', '-m',
        type=str,
        default="FULL",  # 默认值
        choices=["FULL", "SAMPLE"],
        help="验证模式：'FULL' (完整验证所有图像) 或 'SAMPLE' (随机抽样验证)。"
    )
    parser.add_argument(
        '--task', '-t',
        type=str,
        default="segmentation",  # 默认值，根据常见使用情况将默认改为detection
        choices=["detection", "segmentation"],
        help="任务类型：'detection' (目标检测) 或 'segmentation' (实例分割)。"
    )
    # 修改这里的默认参数，使其默认不启用删除
    parser.add_argument(
        '--delete-invalid', '-d',
        action=argparse.BooleanOptionalAction,  # 允许 --delete-invalid / --no-delete-invalid
        default=True,  # 默认不启用删除
        help="是否在验证失败后提供删除不合法图像和标签的选项。默认关闭。使用 --no-delete-invalid 明确禁用。"
    )
    args = parser.parse_args()

    # 从命令行参数获取配置
    VERIFY_MODE = args.mode
    TASK_TYPE = args.task
    ENABLE_DELETE_INVALID = args.delete_invalid  # 是否启用删除功能

    # 设置日志
    verification_logger = setup_logger(base_path=LOGS_DIR,
                                        log_type="dataset_verify",
                                        log_level=logging.INFO,
                                        logger_name="YOLO_DatasetVerification",
                                        )
    verification_logger.info(
        f"当前验证配置：模式='{VERIFY_MODE}', 任务类型='{TASK_TYPE}', 删除非法数据={ENABLE_DELETE_INVALID}")

    # 获取data.yaml文件的路径
    yaml_path = CONFIGS_DIR / "data.yaml"

    # 执行基础验证（图像、标签文件存在性及内容格式）
    verification_logger.info(f"开始基础数据集配置和内容验证 (模式: {VERIFY_MODE})".center(60, '='))
    # verify_dataset_config 现在返回一个元组 (bool, list)
    basic_validation_passed_initial, invalid_data_list = verify_dataset_config(
        yaml_path, verification_logger, mode=VERIFY_MODE, task_type=TASK_TYPE
    )

    # 标记基础验证的问题是否已被“处理”或“解决”
    # 初始如果没有不合法数据，则认为已处理
    basic_validation_problems_handled = basic_validation_passed_initial

    if not basic_validation_passed_initial:  # 如果初始验证未通过（即invalid_data_list非空）
        verification_logger.error("基础数据集验证完成：未通过，请查看详细日志".center(60, '='))

        # 打印不合法数据详情
        verification_logger.error(f"检测到 {len(invalid_data_list)} 个不合法的图像-标签对。详细信息如下：")
        for i, item in enumerate(invalid_data_list):
            verification_logger.error(
                f"  不合法数据 {i + 1}：图像: {item['image_path']}, 标签: {item['label_path']}, 错误: {item['error_message']}"
            )

        # 如果启用了删除功能，则提示用户并提供删除选项
        if ENABLE_DELETE_INVALID:
            # 交互式删除功能，仅在终端运行时提供
            if sys.stdin.isatty():  # 检查是否在交互式终端运行
                print("\n" + "=" * 60)
                print("检测到不合法数据集。是否要删除这些不合法文件？")
                print("注意：删除操作不可逆！")
                print("1. 是，删除图像和对应的标签文件")
                print("2. 否，保留文件")
                print("=" * 60)

                user_choice = input("请输入您的选择 (1 或 2): ")

                if user_choice == '1':
                    delete_invalid_files(invalid_data_list, verification_logger)
                    basic_validation_problems_handled = True  # 用户选择删除，视为问题已被尝试处理
                    verification_logger.info("用户选择删除不合法文件，基础验证问题已尝试处理。")
                elif user_choice == '2':
                    verification_logger.info("用户选择保留不合法文件。")
                    basic_validation_problems_handled = False  # 用户选择不处理，问题未解决
                else:
                    verification_logger.warning("无效选择，不执行删除操作。不合法文件将保留。")
                    basic_validation_problems_handled = False  # 无效选择，问题未解决
            else:  # 在非交互式环境中，如果 ENABLE_DELETE_INVALID 为 True，则自动删除
                verification_logger.warning("在非交互式环境中运行且启用了删除功能。将自动删除不合法文件。")
                delete_invalid_files(invalid_data_list, verification_logger)
                basic_validation_problems_handled = True  # 自动删除，视为问题已被处理
        else:
            verification_logger.info("检测到不合法数据，但未启用删除功能。文件将保留。")
            basic_validation_problems_handled = False  # 未启用删除，问题未解决
    else:  # basic_validation_passed_initial 为 True，即一开始就没有不合法数据
        verification_logger.info(f"基础数据集验证完成：通过".center(60, '='))

    # 执行分割唯一性验证（无论基础验证是否通过，都进行这项检查）
    verification_logger.info(f"开始数据集分割唯一性验证".center(60, '='))
    uniqueness_validation_passed = verify_split_uniqueness(yaml_path, verification_logger)
    if uniqueness_validation_passed:
        verification_logger.info("数据集分割唯一性验证完成：通过".center(60, '='))
    else:
        verification_logger.error("数据集分割唯一性验证完成：未通过，存在重复图像，请查看详细日志".center(60, '='))

    # 总结最终验证结果
    # 只有当基础验证问题得到处理 (或一开始就没有问题) 且唯一性验证通过时，才算全部通过
    if basic_validation_problems_handled and uniqueness_validation_passed:
        verification_logger.info("所有数据集验证均通过！恭喜！".center(60, '='))
    else:
        verification_logger.error("数据集验证未完全通过，请检查以上错误日志！".center(60, '='))