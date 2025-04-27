import os
import time
from functools import lru_cache
from typing import Optional

import cv2
from cv2.typing import MatLike

from zzz_snake_gym import os_utils, cv2_utils


@lru_cache
def get_debug_dir_path() -> str:
    return os_utils.get_path_under_workspace_dir(['.debug'])


@lru_cache()
def get_debug_image_dir_path() -> str:
    return os_utils.get_path_under_workspace_dir(['.debug', 'images'])


def get_debug_image_path(filename, suffix: str = '.png') -> str:
    return os.path.join(get_debug_image_dir_path(), filename + suffix)


def get_debug_image(filename, suffix: str = '.png') -> MatLike:
    return cv2_utils.read_image(get_debug_image_path(filename, suffix))


def save_debug_image(image, file_name: Optional[str] = None, prefix: str = '') -> str:
    if file_name is None:
        file_name = '%s_%d' % (prefix, round(time.time() * 1000))
    path = get_debug_image_path(file_name)
    cv2.imwrite(path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    return file_name
