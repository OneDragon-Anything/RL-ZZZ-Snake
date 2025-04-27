import time
from functools import wraps


def timeit(func):
    """
    用于追踪函数执行时间的装饰器

    示例:
    @timeit
    def my_function():
        # 函数代码
        pass
    """

    @wraps(func)  # 保留原始函数的元数据
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()  # 高精度计时
        result = func(*args, **kwargs)  # 执行原函数
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time

        # 打印函数名和耗时
        print(f"函数 {func.__name__} 执行耗时: {elapsed_time:.6f} 秒")
        return result

    return wrapper