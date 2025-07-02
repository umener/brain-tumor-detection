#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :yolo_predict.py
# @Time      :2025/5/24 14:50
# @Author    :雨霓同学
# @Function  :基于 YOLO 的安全帽检测推理脚本（极简版，支持摄像头/视频流、图像、文件夹、多保存参数、统一输出目录、动态美化参数）

from ultralytics import YOLO
from pathlib import Path
import argparse
import logging
import cv2

import sys

current_path = Path(__file__).parent.parent.resolve()
utils_path = current_path / 'utils'
if str(current_path) not in sys.path:
    sys.path.insert(0, str(current_path))
if str(utils_path) not in sys.path:
    sys.path.insert(1, str(utils_path))

from utils.infer_frame import process_frame
from utils.logging_utils import setup_logger
from utils.config_utils import  log_parameters
from utils.system_utils import log_device_info
from utils.paths import CHECKPOINT_DIR
from utils.config_utils import merge_configs,load_yaml_config

from utils.beautify import BASE_FONT_PATH, TEXT_COLOR, BASE_LINE_WIDTH,calculate_beautify_params
from utils.paths import YOLOSERVER_ROOT
YOLOSERVER_DIR = YOLOSERVER_ROOT
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="基于 YOLO 的安全帽检测推理")
    parser.add_argument('--weights', type=str, default=r'seg_best.pt',
                        help='训练好的模型权重文件（如 best.pt）')
    parser.add_argument('--source', type=str, default='0',
                        help='输入源（图片如 images.jpg，视频如 video.mp4，摄像头如 0）')
    parser.add_argument("--imgsz", type=int, default=640, help="输入图像尺寸")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45, help="IOU 阈值")
    parser.add_argument("--save", type=bool, default=True, help="保存预测结果（图像或视频）")
    parser.add_argument("--save_txt", type=bool, default=False, help="保存预测结果为 TXT")
    parser.add_argument("--save_conf", type=bool, default=False, help="在 TXT 中包含置信度值")
    parser.add_argument("--save_frame", type=bool, default=False, help="保存摄像头/视频每帧图像")
    parser.add_argument("--save_crop", type=bool, default=False, help="保存检测框裁剪图像")
    parser.add_argument("--display-size", type=int, default=720, choices=[360, 720, 1280, 1440], help="摄像头/视频显示分辨率")
    parser.add_argument("--beautify", type=bool, default=True, help="启用美化绘制（圆角标签、中文支持）")
    parser.add_argument("--font-size", type=int, default=26, help="美化字体大小（覆盖自动调整）")
    parser.add_argument("--line-width", type=int, default=4, help="美化线宽（覆盖自动调整）")
    parser.add_argument("--label-padding-x", type=int, default=10, help="美化标签水平内边距（覆盖自动调整）")
    parser.add_argument("--label-padding-y", type=int, default=10, help="美化标签垂直内边距（覆盖自动调整）")
    parser.add_argument("--radius", type=int, default=4, help="美化圆角半径（覆盖自动调整）")
    parser.add_argument("--use-chinese-mapping", type=bool, default=True, help="启用中文映射")
    parser.add_argument("--log_encoding", type=str, default="utf-8", help="日志文件编码")
    parser.add_argument("--use_yaml", type=bool, default=True, help="是否使用 YAML 配置")
    parser.add_argument("--log_level", type=str, default="INFO", help="日志级别（DEBUG/INFO/WARNING/ERROR）")
    return parser.parse_args()


def main():
    """
    主函数，执行 YOLO 模型验证并记录日志。
    """
    args = parse_args()
    # 分辨率映射
    resolution_map = {
        360: (640, 360),
        720: (1280, 720),
        1280: (1920, 1080),
        1440: (2560, 1440),
    }
    display_width, display_height = resolution_map[args.display_size]

    # 美化参数（基值）
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
    print(beautify_params)

    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    model_name = Path(args.weights).stem

    # 初始化日志
    from utils import LOG_DIR
    logger = setup_logger(
        base_path=LOG_DIR,
        log_type="infer",
        model_name=model_name,
        encoding=args.log_encoding,
        log_level=log_level,
        temp_log=False,
    )
    logger.info("===== YOLO 安全帽检测验证开始 =====")

    # 加载 YAML 配置
    yaml_config = {}
    if args.use_yaml:
        yaml_config = load_yaml_config(config_type="infer")

    # 合并命令行和 YAML 参数
    yolo_args, project_args = merge_configs(args, yaml_config, mode="infer")

    # 检查模型文件
    model_path = Path(project_args.weights)
    if not model_path.is_absolute():
        model_path = CHECKPOINT_DIR / project_args.weights
    if not model_path.exists():
        logger.error(f"模型文件不存在: {model_path}")
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    logger.info(f"加载模型: {model_path}")

    # 记录参数、设备和数据集信息
    logger.info("========= 参数信息 =========")
    params_dict = log_parameters(project_args, logger=logger)

    logger.info("========= 设备信息 =========")
    device_dict = log_device_info(logger=logger)

    logger.info("========= 数据集信息 =========")
    logger.info(f"此次使用的数据信息为: {project_args.source}")


    # 加载模型
    logger.info("初始化YOLO模型...")
    model = YOLO(str(model_path))

    # 核心推理
    source = args.source


    # 流式推理（摄像头或视频）
    if source.isdigit() or source.endswith((".mp4", ".avi", ".mov")):
        # 设置显示窗口
        window_name = "YOLO Safety Helmet Detection"
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
        for idx, result in enumerate(model.predict(
            source=source,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            show=False,
            project=YOLOSERVER_DIR / "runs" / "infer",
            name="exp",
            save=args.save,
            save_txt=args.save_txt,
            save_conf=args.save_conf,
            save_crop=args.save_crop,
            # classes = [0,1,3],
            stream=True
        )):
            # 第一帧初始化保存路径
            if idx == 0:
                save_dir = YOLOSERVER_DIR / Path(result.save_dir)
                logger.info(f"此次推理结果保存路径为: {save_dir}")
                if args.save_frame:
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
            # 获取帧
            frame = result.orig_img


            # 处理帧
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

            # 保存帧图像
            if frames_dir:
                cv2.imwrite(str(frames_dir / f"{idx}.jpg"), annotated_frame)

            # 显示
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
        # 非流式推理（图像/文件夹）
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
            project= YOLOSERVER_DIR / "runs" / "infer",
            name="exp",
        )
        # 美化输出
        save_dir = Path(results[0].save_dir)
        bea_save_dir = save_dir / "beautified"
        bea_save_dir.mkdir(parents=True, exist_ok=True)
        print(f"美化参数为：{type(beautify_params)}")
        for ids,result in enumerate(results):
            annotated_frame = process_frame(
                result.orig_img, result, project_args,  beautify_params,
            )

            if args.save:
                cv2.imwrite(str(bea_save_dir/ f"{ids}.jpg"), annotated_frame)
        logger.info(f"推理完成，结果已保存至: {save_dir}")
    logger.info("推理结束".center(80, "="))



if __name__ == "__main__":
    main()
