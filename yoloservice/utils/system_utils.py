#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :system_utils.py
# @Time      :2025/6/12 09:21:57
# @Author    :雨霓同学
# @Project   :SafeH
# @Function  :用于获取设备的硬件信息

import platform
import psutil
import subprocess
import time
import logging
import json
import importlib.metadata  # Python 3.8+ 获取包版本，推荐

import torch
import cpuinfo
import ultralytics
from functools import lru_cache
# 尝试导入ONNX库，如果不存在则设为None，避免程序崩溃
try:
    import onnx
except ImportError:
    onnx = None

# 尝试导入pynvml，用于获取更详细的GPU实时信息。
# 如果pynvml未安装，此模块相关功能将禁用，不会导致程序中断。
_PYNVML_AVAILABLE = False
try:
    from pynvml import *

    _PYNVML_AVAILABLE = True
except ImportError:
    pass  # pynvml is not available, _PYNVML_AVAILABLE remains False


def format_merge(bytes_size: int | float | None):
    """
    格式化内存大小，根据字节数动态选择合适的单位 (GB/MB)。
    能够处理 None 或非数字输入，返回 "N/A"。

    Args:
        bytes_size (int | float | None): 内存大小，单位为字节。

    Returns:
        str: 格式化后的内存大小字符串。
    """
    if bytes_size is None or not isinstance(bytes_size, (int, float)):
        return "N/A"

    # 转换为浮点数以进行安全计算
    bytes_size_float = float(bytes_size)

    if bytes_size_float >= 1024 ** 3:  # 大于等于1GB
        return f"{bytes_size_float / (1024 ** 3):.2f} GB"
    return f"{bytes_size_float / (1024 ** 2):.2f} MB"  # 小于1GB，使用MB


def _get_package_version(package_name: str):
    """
    获取指定Python包的版本号。
    使用 importlib.metadata (Python 3.8+ 及推荐方式)。

    Args:
        package_name (str): 包的名称。

    Returns:
        str: 包的版本号，如果未安装则返回"未安装"。
    """
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "未安装"
    except Exception as e:
        # 捕获其他可能的导入或获取版本异常，记录错误并返回未知
        logging.getLogger("YOLO_Training").warning(f"获取包 '{package_name}' 版本失败: {e}")
        return "获取失败"


def _get_nvidia_driver_version(logger: logging.Logger):
    """
    通过 nvidia-smi 命令获取NVIDIA驱动程序版本。

    Args:
        logger (logging.Logger): 用于记录错误的日志记录器实例。

    Returns:
        str: NVIDIA驱动版本，如果获取失败则返回"获取失败或未安装NVIDIA驱动"。
    """
    try:
        # Use subprocess.run for more robust process management
        results = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,  # Decode stdout/stderr as text
            check=True,  # Raise CalledProcessError for non-zero exit codes
            # Prevent console window from popping up on Windows
            creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
        )
        return results.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        logger.error(f"执行 'nvidia-smi' 命令失败：{e}. (请确保NVIDIA驱动已正确安装并添加到PATH)")
        return "获取失败或未安装NVIDIA驱动"
    except Exception as e:
        logger.error(f"获取NVIDIA驱动程序版本时发生未知错误: {e}")
        return "获取失败或未安装NVIDIA驱动"


def _get_realtime_gpu_metrics(gpu_index: int, logger: logging.Logger):
    """
    使用pynvml获取指定GPU的实时利用率和显存使用情况。

    Args:
        gpu_index (int): GPU的索引。
        logger (logging.Logger): 用于记录错误的日志记录器实例。

    Returns:
        dict: 包含实时GPU指标的字典，如果pynvml不可用或获取失败则返回空字典或错误信息。
    """
    metrics = {}
    if not _PYNVML_AVAILABLE:
        metrics[f"GPU_{gpu_index}_实时使用信息"] = "pynvml库未安装或不可用"
        return metrics

    try:
        handle = nvmlDeviceGetHandleByIndex(gpu_index)
        utilization = nvmlDeviceGetUtilizationRates(handle)
        memory_info = nvmlDeviceGetMemoryInfo(handle)

        metrics[f"GPU_{gpu_index}_利用率"] = f"GPU:{utilization.gpu}% / Mem:{utilization.memory}%"
        metrics[f"GPU_{gpu_index}_实时使用显存"] = format_merge(memory_info.used)
    except NVMLError as error:
        logger.warning(f"获取GPU {gpu_index}实时信息失败(pynvml): {error}")
        metrics[f"GPU_{gpu_index}_实时使用信息"] = "获取失败"
    except Exception as e:
        logger.warning(f"获取GPU {gpu_index}实时信息发生未知错误: {e}")
        metrics[f"GPU_{gpu_index}_实时使用信息"] = "获取失败"
    return metrics


@lru_cache(maxsize=1)
def get_device_info():
    """
    获取当前系统的硬件及环境信息，并以结构化字典形式返回。
    该函数会自动检测CPU/GPU环境，并兼容无GPU的场景。
    使用 @lru_cache 装饰器缓存结果，避免短时间内重复计算。

    Returns:
        dict: 包含系统、CPU、GPU、内存、磁盘和环境信息的字典。
    """
    logger = logging.getLogger("YOLO_Training")

    results = {
        "基本设备信息": {
            "操作系统": f"{platform.system()} {platform.release()}",
            "Python版本": platform.python_version(),
            "Python解释器路径": sys.executable,
            "Python虚拟环境": os.environ.get("CONDA_DEFAULT_ENV", "未知"),
            "当前检测时间": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "主机名": platform.node(),
            "当前用户": os.getenv('USER') or os.getenv('USERNAME', '未知用户'),
        },
        "CPU信息": {
            "CPU型号": cpuinfo.get_cpu_info().get('brand_raw', '未知CPU型号'),
            "CPU物理核心数": psutil.cpu_count(logical=False) or 0,
            "CPU逻辑核心数": psutil.cpu_count(logical=True) or 0,
            "CPU使用率": f"{psutil.cpu_percent()}%",  # 瞬时使用率
        },
        "GPU信息": {},
        "内存信息": {},
        "环境信息": {},
        "磁盘信息": {}
    }

    # --- GPU 信息获取模块 ---
    cuda_available = torch.cuda.is_available()
    _pynvml_local_initialized = False  # Flag to track pynvml initialization inside this function

    if cuda_available:
        gpu_count = torch.cuda.device_count()
        results["GPU信息"] = {
            "CUDA是否可用": True,
            "CUDA版本": torch.version.cuda,
            "NVIDIA驱动程序版本": _get_nvidia_driver_version(logger),  # 调用辅助函数获取驱动版本
            "可用的GPU数量": gpu_count,
        }

        all_gpus_detail = []
        # Attempt to initialize pynvml only if it was successfully imported globally
        if _PYNVML_AVAILABLE:
            try:
                nvmlInit()
                _pynvml_local_initialized = True
            except NVMLError as error:
                logger.warning(f"初始化pynvml失败: {error}. 部分GPU实时信息可能无法获取。")
                _pynvml_local_initialized = False  # Ensure flag is false if init fails

        for i in range(gpu_count):
            try:
                props = torch.cuda.get_device_properties(i)
                gpu_detail = {
                    f"GPU_{i}_型号": torch.cuda.get_device_name(i),
                    f"GPU_{i}_总显存": format_merge(props.total_memory),
                    f"GPU_{i}_算力": f"{props.major}.{props.minor}",
                    f"GPU_{i}_多处理器数量": props.multi_processor_count,
                    f"GPU_{i}_PyTorch_已分配显存": format_merge(torch.cuda.memory_allocated(i)),
                    f"GPU_{i}_PyTorch_已缓存显存": format_merge(torch.cuda.memory_reserved(i)),
                }

                # Get real-time GPU metrics and merge, only if pynvml was successfully initialized
                if _pynvml_local_initialized:
                    gpu_detail.update(_get_realtime_gpu_metrics(i, logger))
                else:
                    gpu_detail[f"GPU_{i}_实时使用信息"] = "pynvml未加载或初始化失败"

                all_gpus_detail.append(gpu_detail)
            except Exception as e:
                logger.error(f"获取GPU {i}详细信息失败: {e}")
                all_gpus_detail.append({f"GPU_{i}_信息": "获取失败或异常"})
        results["GPU详细列表"] = all_gpus_detail

        # Ensure nvml is shut down if it was successfully initialized within this function
        if _pynvml_local_initialized:
            try:
                nvmlShutdown()
            except NVMLError as error:
                logger.warning(f"关闭pynvml失败: {error}")
    else:
        # If CUDA is not available, populate with N/A values
        results["GPU信息"] = {
            "CUDA是否可用": False,
            "CUDA版本": "N/A",
            "NVIDIA驱动程序版本": "N/A",
            "可用的GPU数量": 0,
        }
        results["GPU详细列表"] = {"信息": "未检测到CUDA可用GPU (当前使用CPU)"}

    # --- 内存信息获取模块 ---
    virtual_mem = psutil.virtual_memory()
    results["内存信息"] = {
        "总内存": format_merge(virtual_mem.total),
        "已使用内存": format_merge(virtual_mem.used),
        "剩余内存": format_merge(virtual_mem.available),
        "内存使用率": f"{virtual_mem.percent}%",
    }

    # --- 环境信息获取模块（Python 包版本） ---
    results["环境信息"] = {
        "PyTorch版本": torch.__version__,
        "cuDNN版本": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else "N/A",
        "Ultralytics_Version": ultralytics.__version__,
        "ONNX版本": _get_package_version("onnx"),
        "Numpy版本": _get_package_version("numpy"),
        "OpenCV版本": _get_package_version("opencv-python"),
        "Pillow版本": _get_package_version("Pillow"),
        "Torchvision版本": _get_package_version("torchvision"),
    }

    # --- 磁盘信息获取模块 ---
    disk_info = psutil.disk_usage('/')
    results["磁盘信息"] = {
        "总空间": format_merge(disk_info.total),
        "已用空间": format_merge(disk_info.used),
        "剩余空间": format_merge(disk_info.free),
        "使用率": f"{disk_info.percent}%"
    }

    return results


def format_log_line(key: str, value: str, width: int = 20):
    """
    格式化日志行，使其在控制台输出时中英文对齐。
    通过计算字符的显示宽度（中文字符宽度为2，英文字符宽度为1）进行调整。

    Args:
        key (str): 信息键。
        value (str): 信息值。
        width (int): 目标对齐宽度（英文字符数）。

    Returns:
        str: 格式化后的日志行字符串。
    """
    display_width = sum(2 if 0x4e00 <= ord(char) <= 0x9fff else 1 for char in key)
    padding = width - display_width + len(key)
    return f"    {key:<{padding}}: {value}"


def log_device_info(logger: logging.Logger = None):
    """
    获取并记录设备信息到日志。
    如果未提供logger，则会配置一个简单的控制台logger。

    Args:
        logger (logging.Logger, optional): 用于记录信息的日志记录器实例。
                                        默认为None，此时会使用默认配置。

    Returns:
        dict: 结构化的设备信息字典。
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

    device_info = get_device_info()

    logger.info("=".center(40, '='))
    logger.info("设备信息概览")
    logger.info("=".center(40, '='))

    for category, info in device_info.items():
        if category == "GPU详细列表":
            logger.info(f"{category}:")
            for gpu_idx, gpu_detail in enumerate(info):
                if "未检测到CUDA可用GPU" in gpu_detail.get("信息", ""):
                    logger.info(f"  {gpu_detail['信息']}")
                    break

                logger.info(f"  --- GPU {gpu_idx} 详情 ---")
                for key, value in gpu_detail.items():
                    logger.info(format_log_line(key, value, width=25))
        else:
            logger.info(f"{category}:")
            for key, value in info.items():
                logger.info(format_log_line(key, value, width=20))
    logger.info("=".center(40, '='))
    return device_info


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    result = log_device_info()

    print("\n--- JSON 格式输出 ---")
    print(json.dumps(result, indent=4, ensure_ascii=False))