import traceback
from concurrent.futures import Future, ThreadPoolExecutor

_gym_common_executor = ThreadPoolExecutor(thread_name_prefix='zzz_snake_gym_common', max_workers=16)

def handle_future_result(future: Future):
    try:
        future.result()
    except Exception:
        print('异步执行失败')
        traceback.print_exc()  # 直接打印到标准错误输出


def submit(func, *args, **kwargs) -> Future:
    """
    提交到通用线程池执行
    Args:
        func: 执行函数
        *args: 执行函数的参数
        **kwargs: 执行函数的参数

    Returns:
        Future: 执行结果
    """
    future = _gym_common_executor.submit(func, *args, **kwargs)
    future.add_done_callback(handle_future_result)
    return future