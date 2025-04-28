import os
from functools import lru_cache

from stable_baselines3.common.monitor import Monitor

from zzz_snake_gym.env import ZzzSnakeEnv


@lru_cache
def get_workspace_dir() -> str:
    """

    Returns:
        项目根目录

    """
    dir_path: str = os.path.abspath(__file__)
    for _ in range(2):
        dir_path = os.path.dirname(dir_path)
    return dir_path


def get_path_under_workspace_dir(sub_paths: list[str]) -> str:
    """

    Args:
        sub_paths: 子目录路径 可以传入多个表示多级

    Returns:
        当前工作目录下的子目录路径
    """
    target_path = get_workspace_dir()
    for sub in sub_paths:
        if sub is None:
            continue
        target_path = os.path.join(target_path, sub)
        if not os.path.exists(target_path):
            os.mkdir(target_path)
    return target_path


def get_log_dir_path():
    """

    Returns:
        本项目的workspace目录位置

    """
    dirpath = os.path.abspath(__file__)

    for _ in range(2):
        dirpath = os.path.dirname(dirpath)

    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    return dirpath


def get_tensorboard_log_dir():
    """

    Returns:
        tensorboard日志目录位置

    """
    return get_path_under_workspace_dir(['.log', 'tensorboard'])


def get_sb3_monitor_dir():
    """

    Returns:
        sb3日志目录位置

    """
    return get_path_under_workspace_dir(['.log', 'sb3', 'monitor'])


def get_sb3_checkpoint_dir():
    """

    Returns:
        sb3的checkpoint目录位置
    """
    return get_path_under_workspace_dir(['.log', 'sb3', 'checkpoint'])


def get_sb3_model_save_dir():
    """

    Returns:
        sb3的模型保存目录位置
    """
    return get_path_under_workspace_dir(['.log', 'sb3', 'model'])
