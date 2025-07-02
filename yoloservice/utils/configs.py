#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :configs.py
# @Time      :2025/6/10 21:56:45
# @Author    :雨霓同学
# @Project   :SafeT
# @Function  :

from paths import RUNS_DIR
import torch

DEFAULT_TRAIN_CONFIG = {
    # 基本参数
    'data': 'data.yaml',
    'epochs': 2,
    'time': 'null',  # 明确禁用时间限制
    'batch': 16,
    'imgsz': 640,
    'device': "0" if torch.cuda.is_available() else "cpu",
    'workers': 8,

    # 训练控制
    'patience': 100,
    'save': True,
    'save_period': -1,
    'cache': False,
    'resume': False,
    'amp': True,

    # 项目设置
    'project': str(RUNS_DIR / 'detect'),
    'name': 'train',
    'exist_ok': False,

    # 模型配置
    'pretrained': True,
    'optimizer': 'AdamW',
    'seed': 0,
    'deterministic': True,
    'single_cls': False,
    'classes': None,  # 明确不筛选特定类
    'rect': False,
    'cos_lr': False,
    'multi_scale': False,  # 补充参数

    # 损失权重
    'box': 7.5,
    'cls': 0.5,
    'dfl': 1.5,
    'pose': 12.0,  # 明确不使用姿态估计
    'kobj': 1.0,   # 明确不使用关键点

    # 学习率参数
    'lr0': 0.001,
    'lrf': 0.01,
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 3.0,
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,

    # 数据增强
    'hsv_h': 0.015,
    'hsv_s': 0.7,
    'hsv_v': 0.4,
    'degrees': 0.0,
    'translate': 0.1,
    'scale': 0.5,
    'shear': 0.0,
    'perspective': 0.0,
    'flipud': 0.0,
    'fliplr': 0.5,
    'bgr': 0.0,  # 补充参数
    'mosaic': 1.0,
    'mixup': 0.0,
    'cutmix': 0.0,  # 补充参数
    'copy_paste': 0.0,
    'copy_paste_mode': 'flip',  # 补充参数（无效）
    'auto_augment': 'randaugment',  # 补充参数（无效）
    'erasing': 0.4,

    # 特殊参数
    'close_mosaic': 10,
    'nbs': 64,
    'overlap_mask': True,
    'mask_ratio': 4,
    'dropout': 0.0,
    'val': True,
    'plots': True,
    'profile': False,
    'freeze': None,
    'fraction': 1.0
}

COMMENTED_TRAIN_CONFIG = (
    "# === YOLO 核心训练配置 ===\n"
    "# 注：带(*)参数为高频调整项\n"
    "# 参数参考: https://docs.ultralytics.com/zh/modes/train/#train-settings\n"
    "# 可手动修改参数,或通过命令行进行覆盖如 (--epochs 10) \n"
    "\n"
    "# --- 核心参数 (优先设置) ---\n"
    "# (*)数据集配置文件路径 (YAML格式，需定义train/val路径和类别)\n"
    f"data: {DEFAULT_TRAIN_CONFIG['data']}\n"
    "# (*)训练总轮次 (安全帽检测建议50-300轮)\n"
    f"epochs: {DEFAULT_TRAIN_CONFIG['epochs']}\n"
    "# (*)批量大小 (GPU显存<8G建议8-16，>8G可32-64)\n"
    f"batch: {DEFAULT_TRAIN_CONFIG['batch']}\n"
    "# (*)输入图像尺寸 (安全帽小目标建议>=640)\n"
    f"imgsz: {DEFAULT_TRAIN_CONFIG['imgsz']}\n"
    "# (*)训练设备 (自动选择GPU/CPU，多GPU可用'0,1')\n"
    f"device: '{DEFAULT_TRAIN_CONFIG['device']}'\n"
    "# 是否保存模型和检查点 (True推荐生产环境)\n"
    f"save: {DEFAULT_TRAIN_CONFIG['save']}\n"
    "\n"
    "# --- 训练增强与优化 ---\n"
    "# (*)马赛克增强概率 (小目标检测建议0.75-1.0)\n"
    f"mosaic: {DEFAULT_TRAIN_CONFIG['mosaic']}\n"
    "# (*)水平翻转概率 (安全帽检测推荐0.3-0.7)\n"
    f"fliplr: {DEFAULT_TRAIN_CONFIG['fliplr']}\n"
    "# 垂直翻转概率 (安全帽建议禁用=0)\n"
    f"flipud: {DEFAULT_TRAIN_CONFIG['flipud']}\n"
    "# (*)学习率预热轮次 (通常3-5轮)\n"
    f"warmup_epochs: {DEFAULT_TRAIN_CONFIG['warmup_epochs']}\n"
    "\n"
    "# --- 必要但较少调整的参数 ---\n"
    "# 数据加载线程数 (建议设为CPU核心数的50-75%)\n"
    f"workers: {DEFAULT_TRAIN_CONFIG['workers']}\n"
    "# 早停耐心值 (验证指标无改善的轮次数)\n"
    f"patience: {DEFAULT_TRAIN_CONFIG['patience']}\n"
    "# 是否使用混合精度训练(AMP) (True可加速训练)\n"
    f"amp: {DEFAULT_TRAIN_CONFIG['amp']}\n"
    "# 优化器选择 (SGD/Adam/AdamW/RMSProp等)\n"
    f"optimizer: '{DEFAULT_TRAIN_CONFIG['optimizer']}'\n"
    "\n"
    "# --- 完整参数列表 (按字母排序) ---\n"
    "# 自动增强策略 (仅分类任务有效)\n"
    f"auto_augment: {DEFAULT_TRAIN_CONFIG['auto_augment']}\n"
    "# BGR通道翻转概率\n"
    f"bgr: {DEFAULT_TRAIN_CONFIG['bgr']}\n"
    "# 边界框损失权重\n"
    f"box: {DEFAULT_TRAIN_CONFIG['box']}\n"
    "# 数据缓存方式 (False/ram/disk)\n"
    f"cache: {DEFAULT_TRAIN_CONFIG['cache']}\n"
    "# 分类损失权重\n"
    f"cls: {DEFAULT_TRAIN_CONFIG['cls']}\n"
    "# 关闭马赛克的轮次 (最后N轮)\n"
    f"close_mosaic: {DEFAULT_TRAIN_CONFIG['close_mosaic']}\n"
    "# 是否使用余弦学习率调度\n"
    f"cos_lr: {DEFAULT_TRAIN_CONFIG['cos_lr']}\n"
    "# 复制粘贴增强概率\n"
    f"copy_paste: {DEFAULT_TRAIN_CONFIG['copy_paste']}\n"
    "# 复制粘贴模式\n"
    f"copy_paste_mode: '{DEFAULT_TRAIN_CONFIG['copy_paste_mode']}'\n"
    "# CutMix增强概率\n"
    f"cutmix: {DEFAULT_TRAIN_CONFIG['cutmix']}\n"
    "# 旋转角度范围(度)\n"
    f"degrees: {DEFAULT_TRAIN_CONFIG['degrees']}\n"
    "# 确定性模式 (影响可复现性)\n"
    f"deterministic: {DEFAULT_TRAIN_CONFIG['deterministic']}\n"
    "# DFL损失权重\n"
    f"dfl: {DEFAULT_TRAIN_CONFIG['dfl']}\n"
    "# 随机丢弃率\n"
    f"dropout: {DEFAULT_TRAIN_CONFIG['dropout']}\n"
    "# 是否允许覆盖现有目录\n"
    f"exist_ok: {DEFAULT_TRAIN_CONFIG['exist_ok']}\n"
    "# 使用数据集的比例\n"
    f"fraction: {DEFAULT_TRAIN_CONFIG['fraction']}\n"
    "# 冻结网络层数\n"
    f"freeze: {DEFAULT_TRAIN_CONFIG['freeze']}\n"
    "# 色相增强幅度\n"
    f"hsv_h: {DEFAULT_TRAIN_CONFIG['hsv_h']}\n"
    "# 饱和度增强幅度\n"
    f"hsv_s: {DEFAULT_TRAIN_CONFIG['hsv_s']}\n"
    "# 明度增强幅度\n"
    f"hsv_v: {DEFAULT_TRAIN_CONFIG['hsv_v']}\n"
    "# 关键点对象性损失权重\n"
    f"kobj: {DEFAULT_TRAIN_CONFIG['kobj']}\n"
    "# 初始学习率\n"
    f"lr0: {DEFAULT_TRAIN_CONFIG['lr0']}\n"
    "# 最终学习率比例\n"
    f"lrf: {DEFAULT_TRAIN_CONFIG['lrf']}\n"
    "# 掩码下采样率\n"
    f"mask_ratio: {DEFAULT_TRAIN_CONFIG['mask_ratio']}\n"
    "# MixUp增强概率\n"
    f"mixup: {DEFAULT_TRAIN_CONFIG['mixup']}\n"
    "# 动量参数\n"
    f"momentum: {DEFAULT_TRAIN_CONFIG['momentum']}\n"
    "# 多尺度训练\n"
    f"multi_scale: {DEFAULT_TRAIN_CONFIG['multi_scale']}\n"
    "# 标称批量大小\n"
    f"nbs: {DEFAULT_TRAIN_CONFIG['nbs']}\n"
    "# 训练运行名称\n"
    f"name: '{DEFAULT_TRAIN_CONFIG['name']}'\n"
    "# 掩码重叠处理\n"
    f"overlap_mask: {DEFAULT_TRAIN_CONFIG['overlap_mask']}\n"
    "# 透视变换幅度\n"
    f"perspective: {DEFAULT_TRAIN_CONFIG['perspective']}\n"
    "# 是否生成训练曲线\n"
    f"plots: {DEFAULT_TRAIN_CONFIG['plots']}\n"
    "# 姿态损失权重\n"
    f"pose: {DEFAULT_TRAIN_CONFIG['pose']}\n"
    "# 性能分析模式\n"
    f"profile: {DEFAULT_TRAIN_CONFIG['profile']}\n"
    "# 项目保存目录\n"
    f"project: '{DEFAULT_TRAIN_CONFIG['project']}'\n"
    "# 是否使用预训练权重\n"
    f"pretrained: {DEFAULT_TRAIN_CONFIG['pretrained']}\n"
    "# 矩形训练\n"
    f"rect: {DEFAULT_TRAIN_CONFIG['rect']}\n"
    "# 是否恢复训练\n"
    f"resume: {DEFAULT_TRAIN_CONFIG['resume']}\n"
    "# 检查点保存间隔\n"
    f"save_period: {DEFAULT_TRAIN_CONFIG['save_period']}\n"
    "# 缩放比例\n"
    f"scale: {DEFAULT_TRAIN_CONFIG['scale']}\n"
    "# 随机种子\n"
    f"seed: {DEFAULT_TRAIN_CONFIG['seed']}\n"
    "# 剪切角度\n"
    f"shear: {DEFAULT_TRAIN_CONFIG['shear']}\n"
    "# 单类别模式\n"
    f"single_cls: {DEFAULT_TRAIN_CONFIG['single_cls']}\n"
    "# 最大训练时长(小时)\n"
    f"time: {DEFAULT_TRAIN_CONFIG['time']}\n"
    "# 平移比例\n"
    f"translate: {DEFAULT_TRAIN_CONFIG['translate']}\n"
    "# 是否进行验证\n"
    f"val: {DEFAULT_TRAIN_CONFIG['val']}\n"
    "# 预热偏置学习率\n"
    f"warmup_bias_lr: {DEFAULT_TRAIN_CONFIG['warmup_bias_lr']}\n"
    "# 预热动量\n"
    f"warmup_momentum: {DEFAULT_TRAIN_CONFIG['warmup_momentum']}\n"
    "# 权重衰减\n"
    f"weight_decay: {DEFAULT_TRAIN_CONFIG['weight_decay']}\n"
)

# 默认验证配置
DEFAULT_VAL_CONFIG = {
    'data': 'data.yaml',
    'imgsz': 640,
    'batch': 16,
    'save_json': False,
    'conf': 0.001,
    'iou': 0.7,
    'max_det': 300,
    'half': True,
    'device': '0' if torch.cuda.is_available() else 'cpu',
    'dnn': False,
    'plots': True,
    'classes': None,
    'rect': True,
    'split': 'val',
    'project': str(RUNS_DIR / 'val'),  # 这里控制生成的文件名
    'name': 'exp',   # 这是是每次实验的名称，会逐个叠加expN
    'verbose': False,
    'save_txt': True,
    'save_conf': True,
    'save_crop': True,
    'workers': 8,
    'augment': False,
    'agnostic_nms': False,
    'single_cls': False
}

# 默认推理配置
DEFAULT_INFER_CONFIG = {
    # 基本参数
    'source': '0',  # 指定推理的数据源（图像路径、视频文件、目录、URL 或设备 ID）
    'device': '0' if torch.cuda.is_available() else 'cpu',  # 指定用于推理的设备（例如 cpu, cuda:0 或 0）
    'imgsz': 640,  # 定义用于推理的图像大小（整数或 (高度, 宽度) 元组）
    'batch': 1,  # 指定推理的批量大小（仅当来源为目录、视频文件或 .txt 文件）

    # 模型推理
    'conf': 0.25,  # 设置检测的最小置信度阈值
    'iou': 0.7,  # 非最大抑制 (NMS) 的交叉重叠 (IoU) 阈值
    'max_det': 300,  # 每幅图像允许的最大检测次数
    'classes': None,  # 根据一组类别 ID 过滤预测结果
    'agnostic_nms': False,  # 启用与类别无关的非最大抑制 (NMS)
    'augment': False,  # 可对预测进行测试时间增强 (TTA)
    'half': False,  # 启用半精度 (FP16) 推理
    'stream_buffer': False,  # 决定是否对接收到的视频流帧进行排队
    'vid_stride': 1,  # 视频输入的帧间距
    'retina_masks': False,  # 返回高分辨率分割掩码

    # 保存与项目
    'project': str(RUNS_DIR / 'infer'),  # 保存预测结果的项目目录名称
    'name': 'predict',  # 预测运行的名称
    'save': False,  # 可将注释的图像或视频保存到文件中（Python 中默认为 False）
    'save_frames': False,  # 处理视频时，将单个帧保存为图像
    'save_txt': False,  # 将检测结果保存在文本文件中
    'save_conf': False,  # 在保存的文本文件中包含置信度分数
    'save_crop': False,  # 保存经过裁剪的检测图像
    'stream': False,  # 通过返回结果对象生成器，实现高效内存处理

    # 可视化参数
    'show': False,  # 在窗口中显示注释的图像或视频
    'show_labels': True,  # 在可视输出中显示每次检测的标签
    'show_conf': True,  # 在标签旁显示每次检测的置信度得分
    'show_boxes': True,  # 在检测到的物体周围绘制边框
    'line_width': 8,  # 指定边界框的线宽（None 为自动调整）
    'visualize': False,  # 在推理过程中激活模型特征的可视化
    'verbose': True,  # 控制是否在终端显示详细的推理日志
}

# 带注释的验证配置
COMMENTED_VAL_CONFIG = (
    "# === YOLO 核心验证模式配置 ===\n"
    "# 注：带(*)参数为高频调整项\n"
    "# 参数参考: https://docs.ultralytics.com/zh/modes/val/#arguments-for-yolo-model-validation\n"
    "# 可手动修改参数,或通过命令行进行覆盖如 (--conf 0.5) \n"
    "\n"
    "# --- 核心参数 (优先设置) ---\n"
    "# (*)数据集配置文件路径 (YAML格式，需定义val/test路径和类别)\n"
    f"data: {DEFAULT_VAL_CONFIG['data']}\n"
    "# (*)输入图像尺寸，默认 640\n"
    f"imgsz: {DEFAULT_VAL_CONFIG['imgsz']}\n"
    "# (*)批次大小，默认 16\n"
    f"batch: {DEFAULT_VAL_CONFIG['batch']}\n"
    "# (*)计算设备，0 表示 GPU，cpu 表示 CPU\n"
    f"device: {DEFAULT_VAL_CONFIG['device']}\n"
    "# (*)置信度阈值，默认 0.001\n"
    f"conf: {DEFAULT_VAL_CONFIG['conf']}\n"
    "# (*)IOU 阈值，默认 0.7\n"
    f"iou: {DEFAULT_VAL_CONFIG['iou']}\n"
    "# (*)每幅图像最大检测次数，默认 300\n"
    f"max_det: {DEFAULT_VAL_CONFIG['max_det']}\n"
    "# (*)是否使用半精度推理 (FP16)，默认 True\n"
    f"half: {DEFAULT_VAL_CONFIG['half']}\n"
    "\n"
    "# --- 数据加载参数 ---\n"
    "# 数据加载线程数，默认 8\n"
    f"workers: {DEFAULT_VAL_CONFIG['workers']}\n"
    "# 是否使用测试时间增强 (TTA)，默认 False\n"
    f"augment: {DEFAULT_VAL_CONFIG['augment']}\n"
    "\n"
    "# --- 输出参数 ---\n"
    "# 验证结果保存目录，默认 runs/val\n"
    f"project: {DEFAULT_VAL_CONFIG['project']}\n"
    "# 验证运行名称，默认 exp\n"
    f"name: {DEFAULT_VAL_CONFIG['name']}\n"
    "# 是否保存验证结果为 JSON，默认 False\n"
    f"save_json: {DEFAULT_VAL_CONFIG['save_json']}\n"
    "# 是否保存验证结果为 TXT，默认 False\n"
    f"save_txt: {DEFAULT_VAL_CONFIG['save_txt']}\n"
    "# 是否在 TXT 中包含置信度值，默认 False\n"
    f"save_conf: {DEFAULT_VAL_CONFIG['save_conf']}\n"
    "# 是否保存检测到的物体裁剪图像，默认 False\n"
    f"save_crop: {DEFAULT_VAL_CONFIG['save_crop']}\n"
    "# 是否显示详细验证信息，默认 False\n"
    f"verbose: {DEFAULT_VAL_CONFIG['verbose']}\n"
    "# 是否生成预测结果和混淆矩阵图，默认 False\n"
    f"plots: {DEFAULT_VAL_CONFIG['plots']}\n"
    "\n"
    "# --- 模型参数 ---\n"
    "# 是否使用 OpenCV DNN 推理，默认 False\n"
    f"dnn: {DEFAULT_VAL_CONFIG['dnn']}\n"
    "# 是否使用矩形推理，默认 True\n"
    f"rect: {DEFAULT_VAL_CONFIG['rect']}\n"
    "# 是否将所有类别视为单一类别，默认 False\n"
    f"single_cls: {DEFAULT_VAL_CONFIG['single_cls']}\n"
    "# 是否启用与类别无关的 NMS，默认 False\n"
    f"agnostic_nms: {DEFAULT_VAL_CONFIG['agnostic_nms']}\n"
    "# 指定验证的类 ID 列表，默认 None\n"
    f"classes: {DEFAULT_VAL_CONFIG['classes']}\n"
    "# 数据集分割 (val/test/train)，默认 val\n"
    f"split: {DEFAULT_VAL_CONFIG['split']}\n"
)

COMMENTED_INFER_CONFIG = (
    "# === YOLO 推理模式核心配置 ===\n"
    "# 注：带(*)参数为高频调整项\n"
    "# 参数参考: https://docs.ultralytics.com/zh/modes/predict/#inference-arguments\n"
    "# 可手动修改参数，或通过命令行进行覆盖如 (--conf 0.5)\n"
    "\n"
    "# --- 常见参数 (工地安全帽检测高频调整) ---\n"
    "# (*)数据源，指定工地视频/图像路径、URL或摄像头ID\n"
    f"source: {DEFAULT_INFER_CONFIG['source']}\n"
    "# (*)输入图像尺寸，整数或 (高度, 宽度) 元组，工地场景建议 640 或 1280\n"
    f"imgsz: {DEFAULT_INFER_CONFIG['imgsz']}\n"
    "# (*)置信度阈值，低于此值的检测将被忽略，建议 0.3-0.5 减少误报\n"
    f"conf: {DEFAULT_INFER_CONFIG['conf']}\n"
    "# (*)推理设备，例如 cpu, cuda:0 或 0，None 为自动选择\n"
    f"device: {DEFAULT_INFER_CONFIG['device']}\n"
    "# (*)视频帧采样间隔，1 为每帧处理，增大可跳帧提升速度，默认 1\n"
    f"vid_stride: {DEFAULT_INFER_CONFIG['vid_stride']}\n"
    "# (*)保存注释图像/视频到文件，便于记录违规，Python 中默认 False\n"
    f"save: {DEFAULT_INFER_CONFIG['save']}\n"
    "# (*)保存检测结果为 txt 文件，格式为 [class] [x_center] [y_center] [width] [height] [confidence]，便于分析，默认 False\n"
    f"save_txt: {DEFAULT_INFER_CONFIG['save_txt']}\n"
    "# (*)保存裁剪后的安全帽图像，用于存档或复核，默认 False\n"
    f"save_crop: {DEFAULT_INFER_CONFIG['save_crop']}\n"
    "# (*)实时显示注释图像/视频，适合现场监控或调试，默认 False\n"
    f"show: {DEFAULT_INFER_CONFIG['show']}\n"
    "# (*)启用美化绘制，支持圆角标签和中文显示，适合高质量可视化，默认 True\n"
    f"beautify: {True}\n"
    "# (*)启用中文显示，适合高质量可视化，默认 True\n"
    f"use-chinese: {True}\n"
    "# (*)美化字体大小，以 720p 分辨率为基准，自动缩放，默认 22\n"
    f"font_size: {22}\n"
    "# (*)美化线宽，用于绘制检测框和标签，以 720p 为基准，自动缩放，默认 4\n"
    f"line_width: {4}\n"
    "# (*)美化标签水平内边距，以 720p 为基准，自动缩放，默认 30\n"
    f"label_padding_x: {30}\n"
    "# (*)美化标签垂直内边距，以 720p 为基准，自动缩放，默认 18\n"
    f"label_padding_y: {18}\n"
    "# (*)美化圆角半径，用于标签圆角效果，以 720p 为基准，自动缩放，默认 8\n"
    f"radius: {8}\n"
    "# (*)日志文件编码格式，支持 utf-8-sig、utf-8 等，默认 utf-8-sig\n"
    f"log_encoding: {'utf-8-sig'}\n"
    "# (*)是否使用 YAML 配置文件覆盖命令行参数，适合批量配置，默认 True\n"
    f"use_yaml: {True}\n"
    "# (*)日志级别，支持 DEBUG、INFO、WARNING、ERROR，默认 INFO\n"
    f"log_level: {'INFO'}\n"
    "# (*)额外 YOLO 参数，以键值对形式传递，例如 --key value，默认空列表\n"
    f"extra_args: {[]}\n"
    "\n"
    "# --- 核心参数 ---\n"
    "# 批次大小，仅对目录/视频/txt 文件有效，默认 1\n"
    f"batch: {DEFAULT_INFER_CONFIG['batch']}\n"
    "\n"
    "# --- 模型推理参数 ---\n"
    "# (*)NMS 的 IoU 阈值，控制重叠框的剔除，默认 0.7\n"
    f"iou: {DEFAULT_INFER_CONFIG['iou']}\n"
    "# 每幅图像最大检测次数，默认 300\n"
    f"max_det: {DEFAULT_INFER_CONFIG['max_det']}\n"
    "# 过滤特定类别 ID，例如 [0, 1]，默认 None\n"
    f"classes: {DEFAULT_INFER_CONFIG['classes']}\n"
    "# 与类别无关的 NMS，合并不同类别重叠框，默认 False\n"
    f"agnostic_nms: {DEFAULT_INFER_CONFIG['agnostic_nms']}\n"
    "# 测试时数据增强 (TTA)，提升鲁棒性但降低速度，默认 False\n"
    f"augment: {DEFAULT_INFER_CONFIG['augment']}\n"
    "# 半精度 (FP16) 推理，加速 GPU 推理，默认 False\n"
    f"half: {DEFAULT_INFER_CONFIG['half']}\n"
    "# 视频流帧排队，True 排队不丢帧，False 丢弃旧帧，默认 False\n"
    f"stream_buffer: {DEFAULT_INFER_CONFIG['stream_buffer']}\n"
    "# 返回高分辨率分割掩码，默认 False\n"
    f"retina_masks: {DEFAULT_INFER_CONFIG['retina_masks']}\n"
    "\n"
    "# --- 保存与项目参数 ---\n"
    "# 保存预测结果的项目目录名称，默认 None\n"
    f"project: {DEFAULT_INFER_CONFIG['project']}\n"
    "# 预测运行名称，自动生成子目录，默认 None\n"
    f"name: {DEFAULT_INFER_CONFIG['name']}\n"
    "# 保存视频单帧为图像，默认 False\n"
    f"save_frames: {DEFAULT_INFER_CONFIG['save_frames']}\n"
    "# 在 txt 文件中包含置信度分数，默认 False\n"
    f"save_conf: {DEFAULT_INFER_CONFIG['save_conf']}\n"
    "# 启用流式推理，适合长视频/大量图像，默认 False\n"
    f"stream: {DEFAULT_INFER_CONFIG['stream']}\n"
    "\n"
    "# --- 可视化参数 ---\n"
    "# 显示每次检测的标签，默认 True\n"
    f"show_labels: {DEFAULT_INFER_CONFIG['show_labels']}\n"
    "# (*)显示每次检测的置信度得分，默认 True\n"
    f"show_conf: {DEFAULT_INFER_CONFIG['show_conf']}\n"
    "# 显示检测框，默认 True\n"
    f"show_boxes: {DEFAULT_INFER_CONFIG['show_boxes']}\n"
    "# 检测框线宽，None 为自适应，默认 None\n"
    f"line_width: {DEFAULT_INFER_CONFIG['line_width']}\n"
    "# 激活模型特征可视化，调试用，默认 False\n"
    f"visualize: {DEFAULT_INFER_CONFIG['visualize']}\n"
    "# 显示详细推理日志，默认 True\n"
    f"verbose: {DEFAULT_INFER_CONFIG['verbose']}\n"
)