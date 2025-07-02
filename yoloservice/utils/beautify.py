#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @FileName  : beautify.py
# @Time      : 2025/5/26
# @Author    : 雨霓同学
# @Function  : YOLO 检测结果美化绘制（圆角标签、圆角检测框、中英文支持优化，含特殊圆角处理）

from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from collections import OrderedDict

# ======================= 全局配置（基础值，作为内部默认和参考） ==========================
# 注意：这些是“基准”值，如果用户没有通过CLI/YAML提供，将使用这些默认值
BASE_FONT_PATH = r"D:\BTD\yoloservice\utils\MapleMono550.wght.-VF.ttf"
BASE_FONT_SIZE = 28  # 默认字体大小（针对 REF_DIM 720p 的基准）
BASE_LINE_WIDTH = 4  # 默认线宽
BASE_LABEL_PADDING = (10, 10)  # 默认标签内边距（水平，垂直）
BASE_RADIUS = 4  # 默认圆角半径
TEXT_COLOR = (0, 0, 0)  # 默认文本颜色（BGR，黑色）

LABEL_MAPPING = {
    "glioma_tumor": "胶质瘤",
    "meningioma_tumor": "脑膜瘤",
    "pituitary_tumor": "垂体瘤"
}
COLOR_MAPPING = {
    "glioma_tumor": (255, 0, 0),  # 红色 (BGR)
    "meningioma_tumor": (0, 255, 0),  # 绿色 (BGR)
    "pituitary_tumor": (240, 234, 24)  # 蓝色 (BGR)
}


text_size_cache = OrderedDict()

# 参考分辨率（用于字体大小和线宽的缩放基准）
REF_DIM_FOR_SCALING = 720  # 720p 的短边

# 支持的分辨率映射，用于计算运行时参数
RESOLUTION_DIMS = {
    "360p": (640, 360),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "2K": (2560, 1440),
    "4K": (3840, 2160)
}


# ======================= 参数计算函数（现在接收用户自定义的基准值） ======================
def calculate_beautify_params(img_width, img_height,
                            user_font_path=BASE_FONT_PATH,
                            user_base_font_size=BASE_FONT_SIZE,
                            user_base_line_width=BASE_LINE_WIDTH,
                            user_base_label_padding_x=BASE_LABEL_PADDING[0],
                            user_base_label_padding_y=BASE_LABEL_PADDING[1],
                            user_base_radius=BASE_RADIUS,
                            ref_dim_for_scaling=REF_DIM_FOR_SCALING):
    """
    根据输入图像的原始分辨率和用户提供的基准值，计算实际使用的美化参数。
    同时进行字体缓存的预加载。
    """
    current_short_dim = min(img_width, img_height)
    scale_factor = current_short_dim / ref_dim_for_scaling if ref_dim_for_scaling > 0 else 1.0
    # 使用用户提供的基准值进行缩放
    actual_font_size = max(10, int(user_base_font_size * scale_factor))


    actual_line_width = max(1, int(user_base_line_width * scale_factor))
    actual_label_padding_x = max(5, int(user_base_label_padding_x * scale_factor))
    actual_label_padding_y = max(5, int(user_base_label_padding_y * scale_factor))
    actual_radius = max(3, int(user_base_radius * scale_factor))

    # 预加载字体缓存
    font_sizes_to_preload = generate_preload_font_sizes(
        base_font_size=actual_font_size, # 预加载也基于用户提供的基准字体大小
        ref_dim_for_base_font=ref_dim_for_scaling,
        current_display_short_dim=current_short_dim
    )
    preload_cache(user_font_path, font_sizes_to_preload) # 使用用户提供的字体路径

    return {
        "font_path": user_font_path,
        "font_size": actual_font_size,
        "line_width": actual_line_width,
        "label_padding_x": actual_label_padding_x,
        "label_padding_y": actual_label_padding_y,
        "radius": actual_radius,
        "text_color_bgr": TEXT_COLOR # 文本颜色仍使用全局默认值
    }

# ======================= 美化辅助函数（基于OpenCV） ======================
def draw_filled_rounded_rect(image_np, pt1, pt2, color_bgr, radius,
                            top_left_round=True, top_right_round=True,
                            bottom_left_round=True, bottom_right_round=True):
    """使用 OpenCV 绘制颜色填充的圆角矩形，可控制每个角的圆角状态"""
    x1, y1 = pt1
    x2, y2 = pt2
    thickness = -1

    # 绘制矩形部分 (更精确的绘制，避免在部分圆角为False时出现间隙)
    # 顶部矩形
    cv2.rectangle(image_np,
                (x1 + (radius if top_left_round else 0), y1),
                (x2 - (radius if top_right_round else 0), y1 + radius),
                color_bgr, thickness)
    # 底部矩形
    cv2.rectangle(image_np,
                (x1 + (radius if bottom_left_round else 0), y2 - radius),
                (x2 - (radius if bottom_right_round else 0), y2),
                color_bgr, thickness)
    # 中间矩形
    cv2.rectangle(image_np,
                (x1, y1 + (radius if top_left_round or top_right_round else 0)),
                (x2, y2 - (radius if bottom_left_round or bottom_right_round else 0)),
                color_bgr, thickness)

    # 绘制圆角
    if top_left_round:
        cv2.circle(image_np, (x1 + radius, y1 + radius), radius, color_bgr, thickness, cv2.LINE_AA)
    if top_right_round:
        cv2.circle(image_np, (x2 - radius, y1 + radius), radius, color_bgr, thickness, cv2.LINE_AA)
    if bottom_left_round:
        cv2.circle(image_np, (x1 + radius, y2 - radius), radius, color_bgr, thickness, cv2.LINE_AA)
    if bottom_right_round:
        cv2.circle(image_np, (x2 - radius, y2 - radius), radius, color_bgr, thickness, cv2.LINE_AA)


def draw_bordered_rounded_rect(image_np, pt1, pt2, color_bgr, thickness, radius,
                            top_left_round=True, top_right_round=True,
                            bottom_left_round=True, bottom_right_round=True):
    """使用 OpenCV 绘制带边框的圆角矩形，可控制每个角的圆角状态"""
    x1, y1 = pt1
    x2, y2 = pt2
    line_type = cv2.LINE_AA

    # 绘制直线部分 (更精确的绘制，避免在部分圆角为False时出现间隙)
    # 顶部横线
    cv2.line(image_np,
            (x1 + (radius if top_left_round else 0), y1),
            (x2 - (radius if top_right_round else 0), y1),
            color_bgr, thickness, line_type)
    # 底部横线
    cv2.line(image_np,
            (x1 + (radius if bottom_left_round else 0), y2),
            (x2 - (radius if bottom_right_round else 0), y2),
            color_bgr, thickness, line_type)
    # 左侧竖线
    cv2.line(image_np,
            (x1, y1 + (radius if top_left_round else 0)),
            (x1, y2 - (radius if bottom_left_round else 0)),
            color_bgr, thickness, line_type)
    # 右侧竖线
    cv2.line(image_np,
            (x2, y1 + (radius if top_right_round else 0)),
            (x2, y2 - (radius if bottom_right_round else 0)),
            color_bgr, thickness, line_type)

    # 绘制圆角
    if top_left_round:
        cv2.ellipse(image_np, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color_bgr, thickness, line_type)
    if top_right_round:
        cv2.ellipse(image_np, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color_bgr, thickness, line_type)
    if bottom_left_round:
        cv2.ellipse(image_np, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color_bgr, thickness, line_type)
    if bottom_right_round:
        cv2.ellipse(image_np, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color_bgr, thickness, line_type)


# ======================= 文本和缓存辅助函数（基于Pillow） ======================
def generate_preload_font_sizes(base_font_size, ref_dim_for_base_font, current_display_short_dim, buffer_range=2):
    """
    根据当前显示分辨率和基准字体大小，生成用于预加载的字体大小列表。
    只会预加载当前分辨率下可能用到的字体大小及附近几个值。
    """
    font_sizes_set = set()
    scale_factor = current_display_short_dim / ref_dim_for_base_font if ref_dim_for_base_font > 0 else 1.0
    scaled_base_font_size = int(base_font_size * scale_factor)

    # 加入当前分辨率对应的字体大小，并考虑一些微小的像素偏差
    for i in range(-buffer_range, buffer_range + 1):
        buffered_size = max(10, scaled_base_font_size + i)  # 确保字体大小不小于10
        font_sizes_set.add(buffered_size)
    return sorted(list(font_sizes_set))


def preload_cache(font_path, font_sizes_list):
    """预缓存中英文标签尺寸"""
    global text_size_cache
    text_size_cache.clear()
    for size in font_sizes_list:
        try:
            font = ImageFont.truetype(font_path, size)
        except IOError:
            print(f"警告：无法加载字体文件 '{font_path}'。跳过字体大小 {size} 的预缓存。")
            continue
        # 预缓存中文和英文的典型文本尺寸
        for label_val in list(LABEL_MAPPING.values()) + list(LABEL_MAPPING.keys()):
            text = f"{label_val} 80.0%"  # 使用规范化的置信度字符串
            cache_key = f"{text}_{size}"
            temp_image = Image.new('RGB', (1, 1))
            draw = ImageDraw.Draw(temp_image)
            bbox = draw.textbbox((0, 0), text, font=font)
            text_size_cache[cache_key] = (bbox[2] - bbox[0], bbox[3] - bbox[1])


def get_text_size(text, font_obj, max_cache_size=500):  # 增加缓存大小以适应更多可能的文本
    """计算文本尺寸（带缓存，规范化置信度）"""
    # 尝试提取标签部分来构建规范化的key
    parts = text.split(" ")
    if len(parts) > 1 and parts[-1].endswith('%'):
        label_part = " ".join(parts[:-1])
        # 使用一个规范化的置信度字符串作为缓存key，避免因为置信度细微变化导致缓存命中率低
        normalized_text = f"{label_part} 80.0%"
    else:
        # 如果不是典型的 "Label Confidence%" 格式，则直接使用完整文本作为规范化key
        normalized_text = text

    cache_key = f"{normalized_text}_{font_obj.size}"
    if cache_key in text_size_cache:
        return text_size_cache[cache_key]

    temp_image = Image.new('RGB', (1, 1))
    draw = ImageDraw.Draw(temp_image)
    bbox = draw.textbbox((0, 0), text, font=font_obj)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    text_size_cache[cache_key] = (width, height)
    if len(text_size_cache) > max_cache_size:
        text_size_cache.popitem(last=False)  # 移除最老（最早添加）的项
    return (width, height)


# ======================= 核心绘制函数 ======================
def custom_plot(
        image,
        boxes,
        confs,
        labels,
        use_chinese_mapping=True,
        font_path=BASE_FONT_PATH,
        font_size=BASE_FONT_SIZE,
        line_width=BASE_LINE_WIDTH,
        label_padding_x=BASE_LABEL_PADDING[0],
        label_padding_y=BASE_LABEL_PADDING[1],
        radius=BASE_RADIUS,
        text_color_bgr=TEXT_COLOR,
        beautify=True # 这个参数在这里可以去掉，因为函数名就是custom_plot，就表示要美化
):
    """绘制检测框和标签 (始终执行美化模式)"""
    result_image_cv = image.copy()  # 先在 OpenCV 图像上进行所有非文本绘制
    img_height, img_width = image.shape[:2]
    try:
        font_pil = ImageFont.truetype(font_path, font_size)
    except OSError:
        print(f"错误：无法加载字体文件 '{font_path}'。将使用Pillow默认字体。")
        font_pil = ImageFont.load_default()

    # 存储所有需要绘制的文本信息
    texts_to_draw = []

    for box, conf, label_key in zip(boxes, confs, labels):
        x1, y1, x2, y2 = map(int, box)
        color_bgr = COLOR_MAPPING.get(label_key, (0, 255, 0))  # 默认绿色

        # 标签语言
        if use_chinese_mapping:
            display_label = LABEL_MAPPING.get(label_key, label_key)
        else:
            display_label = label_key
        label_text_full = f"{display_label} {conf * 100:.1f}%"

        # 计算标签框尺寸和文本尺寸
        text_width, text_height = get_text_size(label_text_full, font_pil)
        label_box_actual_width = text_width + 2 * label_padding_x
        label_box_actual_height = text_height + 2 * label_padding_y

        # 确保标签框宽度至少能容纳圆角
        label_box_actual_width = max(label_box_actual_width, 2 * radius)

        # 标签框左对齐检测框
        label_box_x_min = int(x1 - line_width // 2)

        # --- 标签位置决策逻辑优化 ---
        # 默认尝试将标签放在检测框上方（外侧）
        label_box_y_min_potential_above = y1 - label_box_actual_height

        # 标记是否在检测框内部绘制标签
        draw_label_inside = False

        # 如果标签框放在上方会超出图像顶部
        if label_box_y_min_potential_above < 0:
            # 尝试将标签放在检测框内部顶部
            if (y2 - y1) >= (label_box_actual_height + line_width * 2):
                label_box_y_min = int(y1 - line_width / 2)
                label_box_y_max = y1 + label_box_actual_height
                draw_label_inside = True
            else:  # 如果检测框太矮，内部也放不下，则放在检测框下方
                label_box_y_min = y2 + line_width
                label_box_y_max = y2 + label_box_actual_height + line_width
                # 检查是否超出图像底部，如果超出则强制贴底
                if label_box_y_max > img_height:
                    label_box_y_max = img_height
                    label_box_y_min = img_height - label_box_actual_height
                draw_label_inside = False
        else:  # 标签可以正常放在检测框上方（外侧）
            label_box_y_min = label_box_y_min_potential_above
            label_box_y_max = y1
            draw_label_inside = False

        # 标签框水平边界检查
        label_box_x_max = label_box_x_min + label_box_actual_width

        # 定义一个标志，指示标签是否需要靠右对齐检测框
        align_right = False
        if label_box_x_max > img_width:
            align_right = True
            label_box_x_min = int(x2 + line_width // 2) - label_box_actual_width  # 标签框右边界与检测框右边界对齐
            if label_box_x_min < 0:  # 如果右对齐后仍然超出左边界，说明标签框比图像宽
                label_box_x_min = 0

        # 判读标签框宽度是否大于检测框宽度 (影响圆角)
        is_label_wider_than_det_box = label_box_actual_width > (x2 - x1)

        # 定义标签框的圆角状态
        label_top_left_round = True
        label_top_right_round = True
        label_bottom_left_round = True
        label_bottom_right_round = True

        # 根据标签位置和对齐方式调整圆角
        if not draw_label_inside:  # 如果标签在框外
            if label_box_y_min == y1 - label_box_actual_height:  # 标签在检测框上方 (外侧)
                if align_right:  # 如果标签靠右对齐检测框
                    label_bottom_left_round = is_label_wider_than_det_box  # 标签左下角圆角，如果标签比检测框宽
                    label_bottom_right_round = False  # 右下角直角，与检测框右上角对齐
                else:  # 标签靠左对齐检测框 (常规情况或超出左边界)
                    label_bottom_left_round = False  # 底部左角直角，与检测框左上角对齐
                    label_bottom_right_round = is_label_wider_than_det_box  # 右下角圆角，如果标签比检测框宽
            elif label_box_y_min == y2 + line_width:  # 标签在检测框下方 (外侧)
                if align_right:  # 如果标签靠右对齐检测框
                    label_top_left_round = is_label_wider_than_det_box  # 标签左上角圆角，如果标签比检测框宽
                    label_top_right_round = False  # 右上角直角
                else:  # 标签靠左对齐检测框
                    label_top_left_round = False  # 顶部左角直角
                    label_top_right_round = is_label_wider_than_det_box  # 右上角圆角，如果标签比检测框宽
        else:  # 如果标签在检测框内部 (上部贴合)
            label_top_left_round = True
            label_top_right_round = True
            if align_right:  # 如果标签在内部且靠右对齐
                label_bottom_left_round = is_label_wider_than_det_box  # 左下角圆角，如果标签比框宽
                label_bottom_right_round = False  # 右下角直角
            else:  # 标签在内部且靠左对齐
                label_bottom_left_round = False
                # 工况 1: 超上边界，标签框宽度小于检测框时，标签框右下角是圆角矩形。
                label_bottom_right_round = is_label_wider_than_det_box or not is_label_wider_than_det_box  # 如果标签在内部，右下角始终圆角
                # 简化为：
                # label_bottom_right_round = True # 因为在内部时，默认其右下角就是圆角

        # 定义检测框的圆角状态 (基于标签位置)
        det_top_left_round = True
        det_top_right_round = True
        det_bottom_left_round = True
        det_bottom_right_round = True

        if not draw_label_inside:  # 如果标签在框外
            if label_box_y_min == y1 - label_box_actual_height:  # 标签在检测框上方
                if align_right:  # 标签靠右对齐检测框
                    det_top_left_round = is_label_wider_than_det_box  # 如果标签比框宽，检测框左上角为圆角
                    det_top_right_round = False  # 检测框右上角直角，与标签框底部对齐
                else:  # 标签靠左对齐检测框
                    det_top_left_round = False  # 检测框左上角直角，与标签框底部对齐
                    # 工况 2 & 3: 正常情况，标签框宽度大于/小于检测框时，检测框右上角圆角/直角
                    det_top_right_round = not is_label_wider_than_det_box  # 如果标签比框宽，右上角直角；否则圆角
            elif label_box_y_min == y2 + line_width:  # 标签在检测框下方
                if align_right:  # 标签靠右对齐检测框
                    det_bottom_left_round = is_label_wider_than_det_box  # 如果标签比框宽，检测框左下角为圆角
                    det_bottom_right_round = False  # 检测框右下角直角
                else:  # 标签靠左对齐检测框
                    det_bottom_left_round = False  # 检测框左下角直角
                    det_bottom_right_round = is_label_wider_than_det_box  # 如果标签比框宽，检测框右下角为圆角
        else:  # 如果标签在检测框内部 (上部贴合)
            det_top_left_round = False
            det_top_right_round = False

        # 绘制检测框 (OpenCV)
        draw_bordered_rounded_rect(result_image_cv, (x1, y1), (x2, y2),
                                color_bgr, line_width, radius,
                                det_top_left_round, det_top_right_round,
                                det_bottom_left_round, det_bottom_right_round)

        # 绘制填充的标签框
        draw_filled_rounded_rect(result_image_cv, (label_box_x_min, label_box_y_min),
                                (label_box_x_min + label_box_actual_width, label_box_y_max),
                                color_bgr, radius,
                                label_top_left_round, label_top_right_round,
                                label_bottom_left_round, label_bottom_right_round)

        # 文本放置在标签框内居中
        text_x = label_box_x_min + (label_box_actual_width - text_width) // 2
        text_y = label_box_y_min + (label_box_actual_height - text_height) // 2

        # 存储文本信息，稍后统一绘制
        texts_to_draw.append({
            'text': label_text_full,
            'position': (text_x, text_y),
            'font': font_pil,
            'fill_bgr': text_color_bgr
        })

    # 统一绘制所有文本
    if texts_to_draw:
        # 将 OpenCV 图像转换为 Pillow 图像，用于文本绘制
        image_pil = Image.fromarray(cv2.cvtColor(result_image_cv, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(image_pil)

        for text_info in texts_to_draw:
            fill_rgb = (text_info['fill_bgr'][2], text_info['fill_bgr'][1], text_info['fill_bgr'][0])
            draw.text(text_info['position'], text_info['text'], font=text_info['font'], fill=fill_rgb)

        # 将 Pillow 图像转换回 OpenCV 图像
        result_image_cv = cv2.cvtColor(np.array(image_pil), cv2.COLOR_BGR2RGB)

    return result_image_cv