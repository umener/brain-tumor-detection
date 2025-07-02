import torch
from pathlib import Path
from PIL import Image
import numpy as np
from pathlib import Path
import sys
current_path = Path(__file__).parent.parent.resolve()
utils_path = current_path / 'utils'
if str(current_path) not in sys.path:
    sys.path.insert(0, str(current_path))
if str(utils_path) not in sys.path:
    sys.path.insert(1, str(utils_path))
# 假设 CHECKPOINT_DIR 和 process_frame 都在 utils 模块中，且可直接导入
# 请确保 utils 模块在 Python 的搜索路径中，并且 CHECKPOINT_DIR 已正确定义
from utils.paths import CHECKPOINTS_DIR
from utils.infer_frame import process_frame
from ultralytics import YOLO # 将 YOLO 导入移到这里，方便全局使用
# --- 定义 project_args 和 beautify_params ---
class ProjectArgs:
    def __init__(self, beautify=True):
        self.beautify = beautify

DEFAULT_PROJECT_ARGS = ProjectArgs(beautify=True)

DEFAULT_BEAUTIFY_PARAMS = {
    "use_chinese_mapping": True,
    "font_path": r"D:\BTD\yoloservice\utils\LXGWWenKai-Bold.ttf", # <<=== 提醒：请确保这里指向你的字体文件路径
    "font_size": 26,
    "line_width": 4,
    "label_padding_x": 10,
    "label_padding_y": 10,
    "radius": 10,
    "text_color_bgr": (255, 255, 255) # 黑色
}
# --- 结束定义 ---


# --- 全局变量用于存储模型实例 (模拟单例行为) ---
_global_yolo_model = None
_last_loaded_model_path = None
_last_loaded_device = None


def load_yolo_model(model_path: Path, device: str = 'cpu') -> YOLO:
    """
    加载YOLO模型。如果模型已加载且参数相同，则直接返回现有模型实例。

    Args:
        model_path (Path): 你的YOLO模型文件（例如 best.pt）的路径。
        device (str): 推理设备，可以是 'cpu' 或 'cuda'。

    Returns:
        YOLO: 加载的 YOLO 模型实例。
    """
    global _global_yolo_model, _last_loaded_model_path, _last_loaded_device

    # 统一设备字符串
    if device == '0' and torch.cuda.is_available():
        actual_device = 'cuda'
    else:
        actual_device = device

    # 如果模型已加载且路径和设备都相同，则直接返回
    if _global_yolo_model is not None and \
       _last_loaded_model_path == model_path and \
       _last_loaded_device == actual_device:
        print("Using existing YOLO model instance.")
        return _global_yolo_model

    # 否则，加载新模型
    print(f"Loading YOLO model from: {model_path} on device: {actual_device}")
    model = YOLO(model_path)
    model.to(actual_device)
    model.eval() # 设置模型为评估模式
    print("YOLOv8 model loaded successfully.")

    # 更新全局变量
    _global_yolo_model = model
    _last_loaded_model_path = model_path
    _last_loaded_device = actual_device

    return model


def detect_image(image_input_path: str, model: YOLO) -> tuple[list, Image.Image]:
    """
    对输入图片执行YOLO检测，并返回原始检测数据和美化后的图片（PIL.Image对象）。

    Args:
        image_input_path (str): 图片文件路径。
        model (YOLO): 已经加载好的 YOLO 模型实例。

    Returns:
        tuple: (list: 包含检测结果的字典列表, PIL.Image.Image: 美化并绘制了检测结果的图片)
    """
    results = model.predict(image_input_path, conf=0.25, verbose=False)

    detections_data = []
    # 由于不防御性编程，我们假设 results 总是包含有效数据
    r = results[0]

    # 遍历边界框，提取原始检测数据
    for box in r.boxes:
        xmin, ymin, xmax, ymax = map(int, box.xyxy[0].tolist())
        conf = round(box.conf[0].item(), 2)
        cls = int(box.cls[0].item())
        class_name = model.names[cls]  # 获取类别名称

        detections_data.append({
            'box': [xmin, ymin, xmax, ymax],
            'confidence': conf,
            'class_id': cls,
            'class_name': class_name
        })

    # 调用你的 process_frame 函数
    plotted_image_np = process_frame(
        frame=r.orig_img, # 原始图像的 NumPy 数组
        result=r,         # YOLO Results 对象
        project_args=DEFAULT_PROJECT_ARGS,
        beautify_params=DEFAULT_BEAUTIFY_PARAMS
    )

    # 关键修正：将 NumPy 数组转换为 PIL.Image.Image 对象
    # 假设 process_frame 返回的是 BGR 格式的 OpenCV 图像 (NumPy 数组)
    plotted_image_pil = Image.fromarray(plotted_image_np[..., ::-1])

    return detections_data, plotted_image_pil


# --- 辅助函数：获取可用模型 ---
def get_available_models():
    """
    获取 CHECKPOINTS_DIR 中所有可用的模型文件。
    """
    model_dir = CHECKPOINTS_DIR
    model_files = []
    # 这里不进行防御性检查，假设 CHECKPOINT_DIR 存在且是目录
    for f in model_dir.iterdir():
        if f.is_file() and (f.suffix == '.pt' or f.suffix == '.weights'):
            model_files.append(f.name)
    print(f"Found available models: {model_files}")
    return sorted(model_files)


# --- 简单的验证代码块 ---
if __name__ == '__main__':
    print("--- 启动 YOLO 检测器（函数式）验证 ---")

    # --- 1. 获取可用模型 ---
    print("\n尝试获取可用模型:")
    available_models = get_available_models()
    if not available_models:
        print("未找到任何可用模型文件。请检查 CHECKPOINT_DIR 设置和目录内容。")
        exit("无法进行检测，因为没有找到模型。")

    test_model_name = available_models[0] # 假设至少有一个模型
    print(f"选择模型进行测试: {test_model_name}")

    # --- 2. 选择设备 (CPU 或 CUDA) ---
    test_device = 'cpu'
    if torch.cuda.is_available():
        print("CUDA 可用，将使用 CUDA 进行推理。")
        test_device = 'cuda'
    else:
        print("CUDA 不可用，将使用 CPU 进行推理。")

    # --- 3. 加载 YOLO 模型 ---
    model_path_full = CHECKPOINTS_DIR / test_model_name
    yolo_model = load_yolo_model(model_path=model_path_full, device=test_device)

    # --- 4. 准备一张测试图片 ---
    # 请在这里提供你的测试图片路径！
    # 示例路径，请替换为你的实际图片路径
    test_image_path = r"D:\BTD\yoloservice\data\val\images\M_369_jpg.rf.c790215e7a7d20ab02afb85b0d0aee76.jpg"
    print(f"\n准备对图片进行检测: {test_image_path}")

    # --- 5. 执行检测 ---
    detections, plotted_image = detect_image(image_input_path=test_image_path, model=yolo_model)

    print("\n--- 检测结果 ---")
    if detections:
        for det in detections:
            print(f"  类别: {det['class_name']} (ID: {det['class_id']}), "
                  f"置信度: {det['confidence']:.2f}, "
                  f"框: {det['box']}")
    else:
        print("未检测到任何目标。")

    # --- 6. 保存或显示美化后的图片 ---
    output_image_path = Path("detected_image_func_style.jpg")
    plotted_image.save(output_image_path)
    print(f"\n美化后的图片已保存到: {output_image_path.absolute()}")
    # plotted_image.show() # 如果需要显示

    print("\n--- YOLO 检测器（函数式）验证结束 ---")