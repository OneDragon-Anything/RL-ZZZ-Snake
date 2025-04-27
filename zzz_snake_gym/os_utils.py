import json
import os
from functools import lru_cache


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


def save_json(data: dict, file_path: str) -> None:
    """
    保存json文件
    :param data: json数据
    :param file_path: 保存路径
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)