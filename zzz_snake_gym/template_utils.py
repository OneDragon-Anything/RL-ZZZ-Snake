import os

import yaml
from cv2.typing import MatLike

from zzz_snake_gym import cv2_utils
from zzz_snake_gym.match_result import MatchResultList


class TemplateInfo:

    def __init__(self, template_dir: str):
        """

        Args:
            template_dir: 模板目录的绝对路径
        """
        self.raw: MatLike = cv2_utils.read_image(os.path.join(template_dir, 'raw.png'))  # 模板原图
        self.mask: MatLike = cv2_utils.read_image(os.path.join(template_dir, 'mask.png'))  # 模板遮罩

        self.rect: tuple[int, int, int, int] = (0, 0, 0, 0)  # 这个模板应该出现在的位置
        yaml_file_path = os.path.join(template_dir, 'config.yml')
        if os.path.exists(yaml_file_path):
            with open(yaml_file_path, 'r', encoding='utf-8') as file:
                template_info = yaml.safe_load(file)
                point_list = template_info.get('point_list')
                if len(point_list) == 2:
                    lt = point_list[0].split(',')
                    rb = point_list[1].split(',')
                    self.rect = (
                        int(lt[0]),
                        int(lt[1]),
                        int(rb[0]),
                        int(rb[1]),
                    )

        self.to_match_rect: tuple[int, int, int, int] = (
            self.rect[0] - 10,
            self.rect[1] - 10,
            self.rect[2] + 10,
            self.rect[3] + 10,
        )  # 用于匹配的区域


TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')  # 模板目录
TEMPLATE_CACHE: dict[str, TemplateInfo] = {}


def get_template(template_name: str) -> TemplateInfo:
    """
    获取一个模板

    Args:
        template_name: 模板名称

    Returns:
        模板
    """
    global TEMPLATE_CACHE
    template_info = TEMPLATE_CACHE.get(template_name)
    if template_info is None:
        template_dir = os.path.join(TEMPLATE_DIR, template_name)
        template_info = TemplateInfo(template_dir)
        TEMPLATE_CACHE[template_name] = template_info

    return template_info


def match_template_by_config(screenshot: MatLike, template_name: str) -> bool:
    """
    判断游戏画面中是否能匹配特定模板

    Args:
        screenshot: 游戏画面
        template_name: 模板名称

    Returns:
        是否能匹配模板
    """
    template_info = get_template(template_name)
    rect = template_info.to_match_rect
    crop = screenshot[rect[1]: rect[3], rect[0]: rect[2]]
    mrl = cv2_utils.match_template(crop, template_info.raw, threshold=0.7)
    return mrl.max is not None


def match_template(screenshot: MatLike, template_name: str) -> MatchResultList:
    template_info = get_template(template_name)
    mrl = cv2_utils.match_template(screenshot, template_info.raw, threshold=0.7, mask=template_info.mask,
                                   only_best=False)
    return mrl