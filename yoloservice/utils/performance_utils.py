#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :performance_utils.py
# @Time      :2025/6/23 15:37:47
# @Author    :雨霓同学
# @Project   :BTD
# @Function  :放一些性能测试的工具函数
import logging
import time
from functools import wraps


_default_logger = logging.getLogger(__name__)


def time_it(iterations: int = 1, name: str = None, logger_instance=_default_logger):
    """
    一个用于记录函数执行耗时的装饰器函数，实际使用中会传入一个日志记录器
    :param iterations: 函数执行次数，如果大于1，记录平均耗时，等于1，单次执行耗时
    :param name: 用于日志输出的函数类别名称，
    :param logger_instance: 日志记录器实例
    :return:
    """
    _logger_to_use = logger_instance if logger_instance is not None else _default_logger

    # 辅助函数：根据总秒数格式化为最合适的单位
    def _format_time_auto_unit(total_seconds: float) -> str:
        """
        根据总秒数自动选择并格式化为最合适的单位（微秒、毫秒、秒、分钟、小时）。
        """
        if total_seconds < 0.000001:  # 小于1微秒
            return f"{total_seconds * 1_000_000:.3f} 微秒"
        elif total_seconds < 0.001:  # 小于1毫秒
            return f"{total_seconds * 1_000_000:.3f} 微秒"  # 保持微秒精度
        elif total_seconds < 1.0:  # 小于1秒
            return f"{total_seconds * 1000:.3f} 毫秒"
        elif total_seconds < 60.0:  # 小于1分钟
            return f"{total_seconds:.3f} 秒"
        elif total_seconds < 3600:  # 小于1小时
            minutes = int(total_seconds // 60)
            seconds = total_seconds % 60
            return f"{minutes} 分 {seconds:.3f} 秒"
        else:  # 大于等于1小时
            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)
            seconds = total_seconds % 60
            return f"{hours} 小时 {minutes} 分 {seconds:.3f} 秒"

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            func_display_name = name if name is not None else func.__name__
            total_elapsed_time = 0.0
            result = None

            for i in range(iterations):
                start_time = time.perf_counter() # 获取当前时间
                result = func(*args, **kwargs)
                end_time = time.perf_counter()  # 获取结束的时间
                total_elapsed_time += end_time - start_time
            avg_elapsed_time = total_elapsed_time / iterations
            formatted_avg_time = _format_time_auto_unit(avg_elapsed_time)
            if iterations == 1:
                _logger_to_use.info(f"新能测试：'{func_display_name}' 执行耗时: {formatted_avg_time}")
            else:
                _logger_to_use.info(f"性能测试：'{func_display_name}' 执行: {iterations} 次, 单次平均耗时: {formatted_avg_time}")
            return result
        return wrapper
    return decorator


if __name__ == "__main__":
    from logging_utils import setup_logger
    from paths import LOGS_DIR
    logger = setup_logger(base_path=LOGS_DIR, log_type="performance_test")
    @time_it(iterations=5, name="测试函数",logger_instance=logger)
    def test_function():
        time.sleep(0.5)
        print("测试函数执行完成")
    test_function()

