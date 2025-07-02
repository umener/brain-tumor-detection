#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :model_utils.py
# @Time      :2025/6/26 14:13:35
# @Author    :雨霓同学
# @Project   :BTD
# @Function  :模型拷贝

from datetime import  datetime
from pathlib import Path
import shutil
import os
import logging

def copy_checkpoint_models(train_dir,model_filename, checkpoint_dir,logger):
    """
    复制模型到指定的地点
    :param train_dir:
    :param model_filename:
    :param checkpoint_dir:
    :param logger:
    :return:
    """
    if not isinstance(train_dir, Path) or not train_dir.is_dir():
        logger.error(f"{train_dir} 是一个无效的路径")
        return

    if not isinstance(checkpoint_dir, Path) or not checkpoint_dir.is_dir():
        logger.error(f"{checkpoint_dir} 是一个无效的路径,不能存储训练好的模型")
        return

    # 准备准备新的模型文件名
    date_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    base_model_name = Path(model_filename).stem
    train_suffix = train_dir.name

    # 遍历进行复制
    for model_type in ["best", "last"]:
        src_path = train_dir / "weights" / f"{model_type}.pt"
        if src_path.exists():
            checkpoint_name = f"{train_suffix}_{date_str}_{base_model_name}_{model_type}.pt"
            dest_path = checkpoint_dir / checkpoint_name
            try:
                shutil.copy2(src_path, dest_path)
                logger.info(f"{model_type}模型已经从{src_path}复制到至{dest_path}")
            except FileNotFoundError:
                logger.warning(f"{model_type}模型不存在")
            except shutil.SameFileError:
                logger.error(f"源文件和目标文件相同,无法复制")
            except PermissionError:
                logger.error(f"没有权限复制文件")
            except OSError as e:
                logger.error(f"复制文件时出错: {e}")
        else:
            logger.warning(f"{model_type}.pt 模型不存在,不存在预期的源路径")