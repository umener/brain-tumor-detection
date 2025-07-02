#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :yolo_infer.py
# @Time      :2025/6/27 09:52:40
# @Author    :雨霓同学
# @Project   :BTD
# @Function  :模型推理，支持视频，摄像头，文件，多参数保存，统一输出目录，动态美化参数
import argparse
from ultralytics import YOLO
from pathlib import Path
import logging
import cv2

import sys

current_path = Path(__file__).parent.parent.resolve()
utils_path = current_path / 'utils'
if str(current_path) not in sys.path:
    sys.path.insert(0, str(current_path))
if str(utils_path) not in sys.path:
    sys.path.insert(1, str(utils_path))


from utils.infer_frame import process_frame # 处理单帧图像的
from utils.logging_utils import setup_logger, log_parameters
from utils.config_utils import load_config, merge_config
from utils.system_utils import log_device_info
from utils.beautify import (BASE_FONT_PATH, TEXT_COLOR, BASE_LINE_WIDTH, calculate_beautify_params)
from utils.paths import YOLOSERVER_ROOT, LOGS_DIR, CHECKPOINTS_DIR



def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Inference")
    parser.add_argument("--weights", type=str,
            default=r"D:\BTD\yoloservice\models\checkpoints1\train4_20250626-224651_yolo11m_best.pt", help="模型权重信息路径")
    parser.add_argument("--source", type=str,
            default=r"D:\BTD\yoloservice\data\test\images", help="推理数据源")
    parser.add_argument("--imgsz", type=int, default=640, help="推理图片尺寸")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="IOU阈值")
    parser.add_argument("--save", type=bool, default=True, help="保存推理结果")
    parser.add_argument("--save_txt", type=bool, default=True, help="保存推理结果为txt")
    parser.add_argument("--save_conf", type=bool, default=True, help="保存推理结果为txt")
    parser.add_argument("--save_crop", type=bool, default=True, help="保存推理结果为图片")
    parser.add_argument("--save_frames", type=bool, default=True, help="保存推理结果为图片")

    parser.add_argument("--display-size", type=int, default=720, choices=[360,480,720, 1280, 1440],
                        help="摄像头/视频显示分辨率")
    parser.add_argument("--beautify", type=bool, default=True,
                        help="启用美化绘制（圆角标签、中文支持）")
    parser.add_argument("--font-size", type=int, default=26, help="美化字体大小（覆盖自动调整）")
    parser.add_argument("--line-width", type=int, default=4, help="美化线宽（覆盖自动调整）")
    parser.add_argument("--label-padding-x", type=int, default=10, help="美化标签水平内边距（覆盖自动调整）")
    parser.add_argument("--label-padding-y", type=int, default=10, help="美化标签垂直内边距（覆盖自动调整）")
    parser.add_argument("--radius", type=int, default=4, help="美化圆角半径（覆盖自动调整）")
    parser.add_argument("--use-chinese-mapping", type=bool, default=False, help="启用中文映射")
    parser.add_argument("--use_yaml", type=bool, default=True, help="是否使用 YAML 配置")


    return parser.parse_args()

def main():
    args = parse_args()
    resolution_map = {
    360: (640, 360),
    720: (1280, 720),
    1080: (1920, 1080),
    1280: (2560, 1440),
    1440: (3840, 2160)
    }
    display_width, display_height = resolution_map[args.display_size]
    beautify_params = calculate_beautify_params(
        img_width=display_width,
        img_height=display_height,
        user_font_path=BASE_FONT_PATH,
        user_base_font_size=args.font_size,
        user_base_line_width=args.line_width,
        user_base_label_padding_x=args.label_padding_x,
        user_base_label_padding_y=args.label_padding_y,
        user_base_radius=args.radius,
        ref_dim_for_scaling= args.display_size
    )
    beautify_params["use_chinese_mapping"] = args.use_chinese_mapping
    beautify_params["text_color_bgr"] = TEXT_COLOR
    model_name = Path(args.weights).stem

    logger = setup_logger(
        base_path=LOGS_DIR,
        log_type="infer",
        model_name=model_name,
        log_level=logging.INFO,
        temp_log=False,
    )
    # 加载YAML配置
    yaml_config = {}
    if args.use_yaml:
        yaml_config = load_config(config_type="infer")
    # 合并命令行和YAML参数
    yolo_args, project_args = merge_config(args, yaml_config, mode="infer")

    # 检查模型文件
    model_path = Path(project_args.weights)
    if not model_path.is_absolute():
        model_path = CHECKPOINTS_DIR / project_args.weights
    if not model_path.exists():
        logger.error(f"模型文件 '{model_path}' 不存在")
        raise ValueError(f"模型文件 '{model_path}' 不存在")
    logger.info(f"加载推理模型: {project_args.weights}")

    # 记录参数，设备信息，数据集信息
    logger.info("========= 参数信息 =========")
    log_parameters(project_args, logger=logger)
    logger.info("========= 设备信息 =========")
    log_device_info(logger)
    logger.info("========= 数据集信息 =========")
    logger.info(f"此次使用的数据信息为: {project_args.source}")
    logger.info(f"加载推理模型: {project_args.weights}")
    model = YOLO(str(project_args.weights))
    # 核心推理
    source = args.source

    # 流式推理（摄像头或视频）
    if source.isdigit() or source.endswith((".mp4", ".avi", ".mov")):
        # 设置显示窗口
        window_name = "YOLO Tumor Detection"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, display_width, display_height)
        # 初始化视频捕获
        cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开{'摄像头' if source.isdigit() else '视频'}: {source}")
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        # 流式推理
        video_writer = None
        frames_dir = None
        save_dir = None

        for idx,result in enumerate(model.predict(
            source=source,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            show=False,
            project= project_args.project,
            name="exp",
            save=args.save,
            save_txt=args.save_txt,
            save_conf=args.save_conf,
            save_crop=args.save_crop,
            stream=True #开启流模式
        )):
            # 第一帧初始化保存路径
            if idx == 0:
                save_dir = YOLOSERVER_ROOT / Path(result.save_dir)
                logger.info(f"此次推理结果保存路径为: {save_dir}")
                if args.save_frames:
                    frames_dir = save_dir / "0_frames"
                    frames_dir.mkdir(parents=True, exist_ok=True)
                    logger.info(f"帧保存路径为: {frames_dir}")
                if args.save:
                    video_path = save_dir / "output.mp4"
                    logger.info(f"视频保存路径为: {video_path}")
                    video_writer = cv2.VideoWriter(
                        str(video_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (display_width,display_height)
                    )
                    if  video_writer:
                        logger.info("视频写入器已创建")
            # 处理帧
            frame = result.orig_img
            annotated_frame= process_frame(
                frame,
                result,
                project_args,
                beautify_params,
            )
            # 保存视频
            if video_writer:
                annotated_frame = cv2.resize(annotated_frame, (display_width, display_height))
                video_writer.write(annotated_frame)
            cv2.imshow(window_name, annotated_frame)

            # 按 q 或 Esc 退出
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
        # 释放资源
        cap.release()
        if video_writer:
            logger.info("视频写入器已释放资源")
            video_writer.release()
        cv2.destroyAllWindows()
        logger.info(f"{'摄像头' if source.isdigit() else '视频'}推理完成，结果已保存至: {save_dir or '未保存'}")

    else:
        results = model.predict(
            source=source,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            save=args.save,
            save_txt=args.save_txt,
            save_conf=args.save_conf,
            save_crop=args.save_crop,
            show=False,
            project= YOLOSERVER_ROOT / "runs" / "infer",
            name="exp",
        )
        # 美化输出
        save_dir = Path(results[0].save_dir)
        bea_save_dir = save_dir / "beautified"
        bea_save_dir.mkdir(parents=True, exist_ok=True)
        for idx,result in enumerate(results):
            annotated_frame = process_frame(
                result.orig_img, result, project_args, beautify_params
            )
            if args.save:
                cv2.imwrite(str(bea_save_dir/ f"{idx}.png"), annotated_frame)
        logger.info(f"推理完成，结果已保存至: {save_dir}")
    logger.info("推理结束".center(80, "="))


if __name__ == "__main__":
    main()
