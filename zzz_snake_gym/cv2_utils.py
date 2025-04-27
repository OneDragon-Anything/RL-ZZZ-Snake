import os
from typing import Optional, Union

import cv2
import numpy as np
from cv2.typing import MatLike

from zzz_snake_gym.match_result import MatchResultList, MatchResult


def read_image(file_path: str) -> Optional[MatLike]:
    """
    读取图片
    :param file_path: 图片路径
    :return:
    """
    if not os.path.exists(file_path):
        return None
    image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if image.ndim == 2:
        return image
    elif image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif image.ndim == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        return image


def save_image(img: MatLike, file_path: str) -> None:
    """
    保存图片
    :param img: RBG格式的图片
    :param file_path: 保存路径
    """
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(file_path, img)


def show_image(img: MatLike,
               rects: Union[MatchResult, MatchResultList] = None,
               win_name: str = 'DEBUG', wait: Optional[int] = None, destroy_after: bool = False):
    """
    显示一张图片
    :param img: 图片
    :param rects: 需要画出来的框
    :param win_name:
    :param wait: 显示后等待按键的秒数
    :param destroy_after: 显示后销毁窗口
    :return:
    """
    to_show = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    if rects is not None:
        if type(rects) == MatchResult:
            cv2.rectangle(to_show, (rects.x, rects.y), (rects.x + rects.w, rects.y + rects.h), (255, 0, 0), 1)
        elif type(rects) == MatchResultList:
            for i in rects:
                cv2.rectangle(to_show, (i.x, i.y), (i.x + i.w, i.y + i.h), (255, 0, 0), 1)

    cv2.imshow(win_name, to_show)
    if wait is not None:
        cv2.waitKey(wait)
    if destroy_after:
        cv2.destroyWindow(win_name)


def match_template(source: MatLike, template: MatLike, threshold,
                   mask: np.ndarray = None, only_best: bool = True,
                   ignore_inf: bool = False) -> MatchResultList:
    """
    在原图中匹配模板 注意无法从负偏移量开始匹配 即需要保证目标模板不会在原图边缘位置导致匹配不到
    :param source: 原图
    :param template: 模板
    :param threshold: 阈值
    :param mask: 掩码
    :param only_best: 只返回最好的结果
    :param ignore_inf: 是否忽略无限大的结果
    :return: 所有匹配结果
    """
    tx, ty = template.shape[1], template.shape[0]
    # 进行模板匹配
    # show_image(source, win_name='source')
    # show_image(template, win_name='template')
    # show_image(mask, win_name='mask')
    result = cv2.matchTemplate(source, template, cv2.TM_CCOEFF_NORMED, mask=mask)

    match_result_list = MatchResultList(only_best=only_best)
    filtered_locations = np.where(np.logical_and(
        result >= threshold,
        np.isfinite(result) if ignore_inf else np.ones_like(result))
    )  # 过滤低置信度的匹配结果

    # 遍历所有匹配结果，并输出位置和置信度
    for pt in zip(*filtered_locations[::-1]):
        confidence = result[pt[1], pt[0]]  # 获取置信度
        match_result_list.append(MatchResult(confidence, pt[0], pt[1], tx, ty))

    return match_result_list
