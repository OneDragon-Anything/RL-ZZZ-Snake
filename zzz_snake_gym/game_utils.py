from concurrent.futures import Future
from typing import List, Optional

from cv2.typing import MatLike

from zzz_snake_gym import template_utils, cv2_utils, game_const, thread_utils
from zzz_snake_gym.game_const import GAME_RECT, SCORE_Y, SCORE_X_LIST, SCORE_TEMPLATE_NAME
from zzz_snake_gym.template_utils import TemplateInfo


def get_game_part(screenshot: MatLike, copy: bool = False) -> MatLike:
    """
    获取小游戏区域的画面

    Args:
        screenshot: 游戏画面
        copy: 是否需要复制

    Returns:
        小游戏区域的画面
    """
    crop = screenshot[GAME_RECT[1]: GAME_RECT[3], GAME_RECT[0]: GAME_RECT[2]]
    if copy:
        return crop.copy()
    else:
        return crop


def is_game_over(screenshot: MatLike) -> bool:
    """
    判断当前画面是否游戏结束
    最慢会到6ms 后续优化

    Args:
        screenshot: 游戏画面

    Returns:
        是否游戏结束
    """
    if is_in_game(screenshot):
        return template_utils.match_template_by_config(screenshot, 'game_over')
    else:
        return is_game_over_summary_screen(screenshot)


def is_game_over_summary_screen(screenshot: MatLike) -> bool:
    """

    Args:
        screenshot: 游戏画面

    Returns:
        是否游戏结束的结算画面
    """
    return template_utils.match_template_by_config(screenshot, 'over_kill_icon')


def is_in_game(screenshot: MatLike) -> bool:
    """
    判断当前画面是否在游戏内

    Args:
        screenshot: 游戏画面

    Returns:
        是否在游戏内
    """
    return template_utils.match_template_by_config(screenshot, 'start_score')


def get_current_score(screenshot: MatLike, use_async: bool = True) -> int:
    """
    获取当前的游戏分数 大约8ms

    Args:
        screenshot: 游戏画面
        use_async: 使用使用异步匹配

    Returns:
        当前分数
    """
    total: int = 0
    for score_x in SCORE_X_LIST:
        num = get_score_number(
            screenshot,
            (score_x[0] - 5, SCORE_Y[0] - 5, score_x[1] + 5, SCORE_Y[1] + 5),
            use_async=use_async
        )
        total = total * 10 + num
    return total


def get_score_number(screenshot: MatLike, rect: tuple[int, int, int, int], use_async: bool = True) -> int:
    """
    获取指定分数位置的数字

    Args:
        screenshot: 游戏截图
        rect: 指定的分数位置
        use_async: 使用使用异步匹配

    Returns:
        该位置显示的数字
    """
    crop = screenshot[rect[1]: rect[3], rect[0]: rect[2]]
    if use_async:
        future_list: List[Future] = [
            thread_utils.submit(match_score_number, crop, template_utils.get_template(SCORE_TEMPLATE_NAME[i]))
            for i in range(10)
        ]
        num: int = 0
        max_conf: float = 0
        for idx, future in enumerate(future_list):
            conf = future.result()
            if conf > max_conf:
                max_conf = conf
                num = idx
    else:
        num: int = 0
        max_conf: float = 0
        for i in range(10):
            conf = match_score_number(crop, template_utils.get_template(SCORE_TEMPLATE_NAME[i]))
            if conf > max_conf:
                max_conf = conf
                num = i

    return num


def match_score_number(part_screenshot: MatLike, template: TemplateInfo) -> float:
    """
    在指定分数位置的截图上 匹配具体的数字

    Args:
        part_screenshot: 指定分数位置的截图
        template: 需要匹配的数字模板

    Returns:
        置信度
    """
    mrl = cv2_utils.match_template(part_screenshot, template.raw, threshold=0.7)
    if mrl.max is not None:
        return mrl.max.confidence
    else:
        return -1


def is_opposite_direction(d1: int, d2: int) -> bool:
    """
    两个方向是否相反方向
    Args:
        d1: 动作1
        d2: 动作2

    Returns:
        是否相反方向
    """
    if d1 is None or d2 is None:
        return False
    return (
            (d1 == 0 and d2 == 1)
            or (d1 == 1 and d2 == 0)
            or (d1 == 2 and d2 == 3)
            or (d1 == 3 and d2 == 2)
    )


def cal_next_position(head: tuple[int, int], direction: int) -> Optional[tuple[int, int]]:
    """
    计算下一个位置

    Args:
        head: 头部坐标 (y, x)
        direction: 方向

    Returns:
        下一个位置
    """
    if head is None:
        return None
    if direction == 0:
        return head[0] - 1, head[1]
    elif direction == 1:
        return head[0] + 1, head[1]
    elif direction == 2:
        return head[0], head[1] - 1
    elif direction == 3:
        return head[0], head[1] + 1
    else:
        return None


def is_pos_in_grid(pos: tuple[int, int]) -> bool:
    """

    Args:
        pos: 某位置 (y, x)

    Returns:
        某位置是否仍在网格中
    """
    return pos is not None and 0 <= pos[0] < game_const.GRID_ROWS and 0 <= pos[1] < game_const.GRID_COLS