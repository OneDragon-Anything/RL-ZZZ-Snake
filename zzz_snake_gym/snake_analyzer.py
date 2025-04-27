from enum import IntEnum
from typing import Optional, ClassVar

import cv2
import numpy as np
from cv2.typing import MatLike
from numpy._typing import NDArray

import zzz_snake_gym.game_const
from zzz_snake_gym import game_utils, game_const, thread_utils
from zzz_snake_gym.performance_utils import timeit


class ZzzSnakeGameInfo:

    INF_DIS: ClassVar[int] = 100

    def __init__(
            self,
            start_time: float,
            current_time: float,
            screenshot: MatLike,
            game_part: MatLike,
            hsv_game_part: MatLike,

            score: int,
            game_over: bool,
            head: tuple[int, int],
            grid: NDArray[np.uint8],
            can_go_grid_list: list[NDArray[np.uint8]],
            total_reward_cnt: int,
    ):
        self.game_round: str = str(int(start_time))  # 游戏轮次标志 使用时间为标志
        self.start_time: float = start_time
        self.current_time: float = current_time
        self.survival_seconds: float = current_time - start_time

        # 游戏画面部分
        self.screenshot: MatLike = screenshot  # 整个绝区零的游戏画面
        self.game_part: MatLike = game_part  # 蛇对蛇部分的游戏画面
        self.hsv_game_part: MatLike = hsv_game_part  # 蛇对蛇部分的游戏画面 HSV格式
        self.score: int = score  # 当前画面显示的分数
        self.game_over: bool = game_over  # 当前画面是否在游戏结束

        self.head: tuple[int, int] = head
        self.grid: NDArray[np.uint8] = grid
        self.can_go_grid_list: list[NDArray[np.uint8]] = can_go_grid_list

        self.can_go_cnt: list[int] = []  # 各方向可前往的格子数量
        self.can_go_reward_cnt: list[int] = []  # 各方向可前往的奖励格子数量
        self.direction_danger: list[int] = []  # 各方向的下一步是否危险
        self.direction_reward: list[int] = []  # 各方向的下一步是否奖励
        self.total_reward_cnt: int = max(total_reward_cnt, 1)
        self.cal_by_direction()

        self.direction: int = -1  # 在分析的时候还没有做出动作 方向未知
        self.real_direction: int = -1  # 真实的方向 即当前动作可能不生效 需要保持上一个方向
        self.last_real_direction: int = -1  # 上一个真实的方向 当前真实方向朝向障碍或墙边 其实无法产生移动 因此要用上一个真实方向继续判断当前的动作方向是否合法
        self.predict_head: Optional[tuple[int, int]] = None

        self.dis_to_reward: int = ZzzSnakeGameInfo.INF_DIS
        self.closest_reward_dy_1: int = -game_const.GRID_ROWS
        self.closest_reward_dy_2: int = game_const.GRID_ROWS
        self.closest_reward_dx_1: int = -game_const.GRID_COLS
        self.closest_reward_dx_2: int = game_const.GRID_COLS
        self.cal_reward_dis()

        self.in_boundary: bool = False  # 当前位置是否在边界上
        self.dis_to_boundary: int = ZzzSnakeGameInfo.INF_DIS  # 与边界的距离

        # 根据坐标进行的一系列判断
        self.is_towards_boundary: bool = False
        self.is_predict_danger: bool = False
        self.is_predict_reward: bool = False  # 预估下一步到达食物

    def set_direction(self, direction: int, last_info: Optional['ZzzSnakeGameInfo']) -> None:
        """
        设置本次执行的动作 同时预测下一个位置
        Args:
            direction: 动作
            last_info: 上一个游戏信息

        Returns:

        """
        # 非法的方向
        if direction < 0 or direction > 3:
            return

        self.direction = direction
        last_direction = last_info.real_direction if last_info is not None else direction

        if game_utils.is_opposite_direction(direction, last_direction):
            # 相反操作无效 继续使用之前的操作
            # 不能合并到下述的转向蛇身判断 因为长度1的时候也不能反方向移动 这时候反方向没有蛇身
            self.real_direction = last_direction
        else:
            predict_pos = game_utils.cal_next_position(self.head, direction)
            if (predict_pos is not None
                and game_utils.is_pos_in_grid(predict_pos)
                and self.grid[predict_pos[0], predict_pos[1]] == game_const.GridType.OWN_BODY
            ):
                # 如果是在尝试转向蛇身 则该操作肯定非法 需要保持使用上一个真实方向
                # 主要出现在靠墙移动时 先转向墙 再想转向前一个方向的反方向
                # 例如在最上方往右移动 先按上 再按左 则最后的左操作无效
                self.real_direction = last_direction
            else:
                self.real_direction = direction

        # 计算下一个位置 和其它标记位信息
        if self.head is not None:
            self.predict_head = game_utils.cal_next_position(self.head, self.real_direction)
            self.cal_boundary_dis()
            self.is_predict_danger = self.cal_is_predict_danger()
            self.is_predict_reward = self.cal_is_predict_reward()

    def cal_boundary_dis(self) -> None:
        """
        计算边界相关的距离
        Returns:
            None
        """
        if self.head is not None:
            if self.real_direction == 0:
                self.dis_to_boundary = self.head[0]
            elif self.real_direction == 1:
                self.dis_to_boundary = zzz_snake_gym.game_const.GRID_ROWS - 1 - self.head[0]
            elif self.real_direction == 2:
                self.dis_to_boundary = self.head[1]
            elif self.real_direction == 3:
                self.dis_to_boundary = zzz_snake_gym.game_const.GRID_COLS - 1 - self.head[1]

            if self.head[0] == 0 or self.head[0] == zzz_snake_gym.game_const.GRID_ROWS - 1:
                self.in_boundary = True
            elif self.head[1] == 0 or self.head[1] == zzz_snake_gym.game_const.GRID_COLS - 1:
                self.in_boundary = True
            else:
                self.in_boundary = False

    def cal_is_predict_danger(self) -> bool:
        """
        Returns:
            当前动作是否朝未知格子前进
        """
        # 预估位置非法
        if not game_utils.is_pos_in_grid(self.predict_head):
            return True

        return 0 <= self.real_direction < len(self.direction_danger) and self.direction_danger[self.real_direction] == 1

    def cal_reward_dis(self) -> None:
        """
        计算最近的奖励点距离及其偏移量

        Args:
            head: 当前头部坐标 (row, col)

        Returns:
            元组 (最小距离, dy, dx)
            如果没有奖励点，返回 (INF_DIS, 0, 0)
        """
        if self.head is None:
            return

        # 创建奖励点的布尔掩码
        reward_mask = ((self.grid == game_const.GridType.YELLOW_CRYSTAL) |
                       (self.grid == game_const.GridType.GREEN_SPEED) |
                       (self.grid == game_const.GridType.BLUE_DIAMOND))

        # 获取所有奖励点的坐标
        reward_positions = np.argwhere(reward_mask)

        if len(reward_positions) == 0:
            return

        # 计算所有奖励点到头部的偏移量
        offsets = reward_positions - np.array(self.head)

        # 计算曼哈顿距离
        distances = np.abs(offsets).sum(axis=1)

        # 找到最小距离的索引
        min_idx = np.argmin(distances)

        # 获取最小距离和对应的偏移量
        min_distance = distances[min_idx]
        dy, dx = offsets[min_idx]

        # 找出 dy 和 dx 的最小值和最大值
        min_dy = offsets[:, 0].min()
        max_dy = offsets[:, 0].max()
        min_dx = offsets[:, 1].min()
        max_dx = offsets[:, 1].max()

        self.dis_to_reward = min_distance
        if min_dy < 0:
            self.closest_reward_dy_1 = min_dy
        if max_dy > 0:
            self.closest_reward_dy_2 = max_dy
        if min_dx < 0:
            self.closest_reward_dx_1 = min_dx
        if max_dx > 0:
            self.closest_reward_dx_2 = max_dx

    def cal_by_direction(self) -> None:
        """
        计算4个方向将会遇到的情况
        """
        for direction in range(4):
            if self.head is None:
                self.can_go_cnt.append(0)
                self.can_go_reward_cnt.append(0)
                self.direction_danger.append(1)
                self.direction_reward.append(0)
                continue

            can_go_grid = self.can_go_grid_list[direction]

            # 可前往的下标
            can_go_idx = (can_go_grid == 2)
            # 奖励的下标
            reward_idx = (
                (self.grid == game_const.GridType.YELLOW_CRYSTAL)
                | (self.grid == game_const.GridType.GREEN_SPEED)
                | (self.grid == game_const.GridType.BLUE_DIAMOND)
            )

            can_go_y, can_go_x = np.where(can_go_idx)
            reward_y, reward_x = np.where(can_go_idx & reward_idx)

            self.can_go_cnt.append(len(can_go_x))
            self.can_go_reward_cnt.append(len(reward_x))

            next_pos = game_utils.cal_next_position(self.head, direction)
            if not game_utils.is_pos_in_grid(next_pos):  # 出界了
                self.direction_danger.append(1)
                self.direction_reward.append(0)
            else:
                next_grid_type = self.grid[next_pos[0], next_pos[1]]
                if next_grid_type == game_const.GridType.EMPTY:
                    self.direction_danger.append(0)
                    self.direction_reward.append(0)
                elif (
                    next_grid_type == game_const.GridType.YELLOW_CRYSTAL
                    or next_grid_type == game_const.GridType.BLUE_DIAMOND
                    or next_grid_type == game_const.GridType.GREEN_SPEED
                ):  # 奖励格子
                    self.direction_danger.append(0)
                    self.direction_reward.append(1)
                else:  # 剩余都是不可以移动的位置
                    self.direction_danger.append(1)
                    self.direction_reward.append(0)

    def cal_is_predict_reward(self) -> bool:
        return self.real_direction is not None and self.direction == self.real_direction and self.direction_reward[self.real_direction]


class ZzzSnakeAnalyzer:

    def __init__(self, target_size: tuple[int, int]):
        self.target_size = target_size  # 最后输出的尺寸 高、宽

    def analyse(self, screenshot: MatLike, current_time: float, last_info: ZzzSnakeGameInfo) -> ZzzSnakeGameInfo:
        """
        平均22ms
        Args:
            screenshot: 整个游戏画面
            current_time: 截图时间
            last_info: 上一次的分析结果

        Returns:

        """
        # score: int = game_utils.get_current_score(screenshot)
        score = 0  # 统计分数感觉意义不大 只要存活下去分数自然高
        game_over: bool = game_utils.is_game_over(screenshot)

        # 游戏区域
        game_part = game_utils.get_game_part(screenshot, copy=False)
        if self.target_size[0] != game_part.shape[0] or self.target_size[1] != game_part.shape[1]:
            game_part = cv2.resize(game_part, self.target_size, interpolation=cv2.INTER_AREA)
        hsv_game_part = cv2.cvtColor(game_part, cv2.COLOR_RGB2HSV)

        # 使用HSV筛选一些区域mask
        empty_future = thread_utils.submit(cv2.inRange, hsv_game_part, game_const.EMPTY_HSV_RANGE[0], game_const.EMPTY_HSV_RANGE[1])
        own_head_future = thread_utils.submit(cv2.inRange, hsv_game_part, game_const.OWN_HEAD_HSV_RANGE[0], game_const.OWN_HEAD_HSV_RANGE[1])
        own_body_future = thread_utils.submit(cv2.inRange, hsv_game_part, game_const.OWN_BODY_HSV_RANGE[0], game_const.OWN_BODY_HSV_RANGE[1])
        blue_head_future = thread_utils.submit(cv2.inRange, hsv_game_part, game_const.BLUE_HEAD_HSV_RANGE[0], game_const.BLUE_HEAD_HSV_RANGE[1])
        blue_body_future = thread_utils.submit(cv2.inRange, hsv_game_part, game_const.BLUE_BODY_HSV_RANGE[0], game_const.BLUE_BODY_HSV_RANGE[1])
        purple_head_future = thread_utils.submit(cv2.inRange, hsv_game_part, game_const.PURPLE_HEAD_HSV_RANGE[0], game_const.PURPLE_HEAD_HSV_RANGE[1])
        purple_body_future = thread_utils.submit(cv2.inRange, hsv_game_part, game_const.PURPLE_BODY_HSV_RANGE[0], game_const.PURPLE_BODY_HSV_RANGE[1])
        pink_head_future = thread_utils.submit(cv2.inRange, hsv_game_part, game_const.PINK_HEAD_HSV_RANGE[0], game_const.PINK_HEAD_HSV_RANGE[1])
        pink_body_future = thread_utils.submit(cv2.inRange, hsv_game_part, game_const.PINK_BODY_HSV_RANGE[0], game_const.PINK_BODY_HSV_RANGE[1])
        yellow_crystal_future = thread_utils.submit(cv2.inRange, hsv_game_part, game_const.YELLOW_CRYSTAL_HSV_RANGE[0], game_const.YELLOW_CRYSTAL_HSV_RANGE[1])
        green_speed_future = thread_utils.submit(cv2.inRange, hsv_game_part, game_const.GREEN_SPEED_HSV_RANGE[0], game_const.GREEN_SPEED_HSV_RANGE[1])
        blue_diamond_future = thread_utils.submit(cv2.inRange, hsv_game_part, game_const.BLUE_DIAMOND_HSV_RANGE[0], game_const.BLUE_DIAMOND_HSV_RANGE[1])
        bomb_future = thread_utils.submit(cv2.inRange, hsv_game_part, game_const.BOMB_HSV_RANGE[0], game_const.BOMB_HSV_RANGE[1])

        empty_mask = empty_future.result()
        own_head_mask = own_head_future.result()
        own_body_mask = own_body_future.result()
        blue_head_mask = blue_head_future.result()
        blue_body_mask = blue_body_future.result()
        purple_head_mask = purple_head_future.result()
        purple_body_mask = purple_body_future.result()
        pink_head_mask = pink_head_future.result()
        pink_body_mask = pink_body_future.result()
        yellow_crystal_mask = yellow_crystal_future.result()
        green_speed_mask = green_speed_future.result()
        blue_diamond_mask = blue_diamond_future.result()
        bomb_mask = bomb_future.result()

        empty_future = thread_utils.submit(get_coordinate_by_mask_center, empty_mask, 2, 'avg', 125)
        own_head_future = thread_utils.submit(get_coordinate_by_mask_center, own_head_mask, 1, 'avg', 255)
        own_body_future = thread_utils.submit(get_coordinate_by_mask_center, own_body_mask, 2, 'avg', 255)
        blue_head_future = thread_utils.submit(get_coordinate_by_mask_center, blue_head_mask, 2, 'avg', 255)
        blue_body_future = thread_utils.submit(get_coordinate_by_mask_center, blue_body_mask, 2, 'avg', 255)
        purple_head_future = thread_utils.submit(get_coordinate_by_mask_center, purple_head_mask, 2, 'avg', 255)
        purple_body_future = thread_utils.submit(get_coordinate_by_mask_center, purple_body_mask, 2, 'avg', 255)
        pink_head_future = thread_utils.submit(get_coordinate_by_mask_center, pink_head_mask, 2, 'avg', 255)
        pink_body_future = thread_utils.submit(get_coordinate_by_mask_center, pink_body_mask, 2, 'avg', 255)
        yellow_crystal_future = thread_utils.submit(get_coordinate_by_mask_center, yellow_crystal_mask, 2, 'avg', 125)
        green_speed_future = thread_utils.submit(get_coordinate_by_mask_center, green_speed_mask, 2, 'avg', 125)
        blue_diamond_future = thread_utils.submit(get_coordinate_by_mask_center, blue_diamond_mask, 2, 'avg', 125)
        bomb_future = thread_utils.submit(get_coordinate_by_mask_center, bomb_mask, 5, 'avg', 10)

        empty_list = empty_future.result()
        own_head_list = own_head_future.result()
        own_body_list = own_body_future.result()
        blue_head_list = blue_head_future.result()
        blue_body_list = blue_body_future.result()
        purple_head_list = purple_head_future.result()
        purple_body_list = purple_body_future.result()
        pink_head_list = pink_head_future.result()
        pink_body_list = pink_body_future.result()
        yellow_crystal_list = yellow_crystal_future.result()
        green_speed_list = green_speed_future.result()
        blue_diamond_list = blue_diamond_future.result()
        bomb_list = bomb_future.result()

        head, grid, can_go_grid_list = make_grid(
            last_info=last_info,
            empty_list=empty_list,
            own_head_list=own_head_list,
            own_body_list=own_body_list,
            blue_head_list=blue_head_list,
            blue_body_list=blue_body_list,
            purple_head_list=purple_head_list,
            purple_body_list=purple_body_list,
            pink_head_list=pink_head_list,
            pink_body_list=pink_body_list,
            yellow_crystal_list=yellow_crystal_list,
            green_speed_list=green_speed_list,
            blue_diamond_list=blue_diamond_list,
            bomb_list=bomb_list,
        )
        total_reward_cnt: int = len(yellow_crystal_list[0]) + len(green_speed_list[0]) + len(blue_diamond_list[0])

        info = ZzzSnakeGameInfo(
            start_time=current_time if last_info is None else last_info.start_time,
            current_time=current_time,
            screenshot=screenshot,
            game_part=game_part,
            hsv_game_part=hsv_game_part,
            score=score,
            game_over=game_over,
            head=head,
            grid=grid,
            can_go_grid_list=can_go_grid_list,
            total_reward_cnt=total_reward_cnt,
        )

        return info


def get_coordinate_by_mask_center(
        mask: MatLike,
        radius: int = 1,
        sample_way: str = 'avg',
        lower: int = 255
) -> tuple[list[int], list[int]]:
    """
    从掩码图上 对网格中点采样 并输出符合条件的网格坐标

    Args:
        mask: 掩码图
        radius: 网格中点的采样半径 默认1表示3x3区域
        sample_way: 采样方式 avg=平均值 max=最大值
        lower: 采样后 大于等于多少的值 就认为是符合的坐标

    Returns:
        coordinate_list: 符合条件的坐标列表 [(y, x), ...]
    """
    grid_mask = get_grid_center_mask(mask, radius=radius, sample_way=sample_way)
    return np.where(grid_mask >= lower)


def get_grid_center_mask(mask: NDArray[np.uint8], radius: int = 0, sample_way: str = 'avg') -> NDArray[np.uint8]:
    """
    将H*W的mask分割成网格并提取每个格子中心附近区域的平均值（向量化实现）

    Args:
        mask: 输入的H*W mask数组
        radius: 中心点周围取样的半径，默认1表示3x3区域
        sample_way: 采样方式 avg=平均值 max=最大值
    Returns:
        grid_values: (25, 29) 的数组（根据GRID_ROWS和GRID_COLS）
    """
    h, w = mask.shape  # 现在只有高度和宽度，没有通道

    # 计算每个格子的高度和宽度
    cell_h = h * 1.0 / zzz_snake_gym.game_const.GRID_ROWS
    cell_w = w * 1.0 / zzz_snake_gym.game_const.GRID_COLS

    # 计算所有格子的中心坐标 (y, x)
    y_centers = np.round((np.arange(zzz_snake_gym.game_const.GRID_ROWS) * cell_h) + (cell_h / 2.0)).astype(int)
    x_centers = np.round((np.arange(zzz_snake_gym.game_const.GRID_COLS) * cell_w) + (cell_w / 2.0)).astype(int)

    # 创建取样区域的偏移量网格
    y_offsets, x_offsets = np.meshgrid(np.arange(-radius, radius + 1),
                                       np.arange(-radius, radius + 1))

    # 计算所有取样点的坐标 (grid_rows, grid_cols, (2r+1)^2, 2)
    y_samples = y_centers[:, None, None, None] + y_offsets[None, None, :, :]
    x_samples = x_centers[None, :, None, None] + x_offsets[None, None, :, :]

    # 确保坐标不越界
    y_samples = np.clip(y_samples, 0, h - 1)
    x_samples = np.clip(x_samples, 0, w - 1)

    # 提取所有取样点的值 (grid_rows, grid_cols, (2r+1)^2)
    samples = mask[y_samples.astype(int), x_samples.astype(int)]

    # 计算每个网格的平均值 (grid_rows, grid_cols)
    if sample_way == 'avg':
        return np.mean(samples, axis=(2, 3))
    else:
        return np.max(samples, axis=(2, 3))


def get_coordinate_in_grid_mask(mask: MatLike, lower: int = 255) -> list[tuple[int, int]]:
    """
    根据范围值 获取指定范围内的坐标

    Args:
        mask: 网格掩码
        lower: 范围下限

    Returns:
        符合范围的坐标 (y, x)
    """
    # 获取坐标
    y_coords, x_coords = np.where(mask >= lower)
    # 将坐标组合成 (y, x) 对
    return [(int(y), int(x)) for x, y in zip(x_coords, y_coords)]


def make_grid(
        last_info: ZzzSnakeGameInfo,
        empty_list: tuple[list[int], list[int]],
        own_head_list: tuple[list[int], list[int]],
        own_body_list: tuple[list[int], list[int]],
        blue_head_list: tuple[list[int], list[int]],
        blue_body_list: tuple[list[int], list[int]],
        purple_head_list: tuple[list[int], list[int]],
        purple_body_list: tuple[list[int], list[int]],
        pink_head_list: tuple[list[int], list[int]],
        pink_body_list: tuple[list[int], list[int]],
        yellow_crystal_list: tuple[list[int], list[int]],
        green_speed_list: tuple[list[int], list[int]],
        blue_diamond_list: tuple[list[int], list[int]],
        bomb_list: tuple[list[int], list[int]],
) -> tuple[tuple[int, int], NDArray[np.uint8], list[NDArray[np.uint8]]]:
    grid = np.full((game_const.GRID_ROWS, game_const.GRID_COLS),
                   game_const.GridType.UNKNOWN, dtype=np.int8)

    grid[empty_list[0], empty_list[1]] = game_const.GridType.EMPTY
    grid[own_body_list[0], own_body_list[1]] = game_const.GridType.OWN_BODY
    # 结合颜色匹配和上一次预测的位置 筛选真实的head
    head = None
    # 先使用颜色匹配 找蛇身颜色范围能匹配的蛇头
    for y, x in zip(own_head_list[0], own_head_list[1]):
        if grid[y][x] == game_const.GridType.OWN_BODY:
            if head is None:
                head = (int(y), int(x))
            else:
                head = None
                break

    # 使用颜色匹配不到的时候 使用上一次预测的位置
    if (head is None and last_info is not None
            and last_info.predict_head is not None
            and game_utils.is_pos_in_grid(last_info.predict_head)
            and grid[last_info.predict_head[0]][last_info.predict_head[1]] == game_const.GridType.OWN_BODY
    ):
        head = last_info.predict_head
    # 如果上一次预测的位置不合法 则使用上一次的真实位置
    if (head is None and last_info is not None
            and last_info.head is not None
            and game_utils.is_pos_in_grid(last_info.head)
            and grid[last_info.head[0]][last_info.head[1]] == game_const.GridType.OWN_BODY
    ):
        head = last_info.head

    if head is not None:
        grid[head[0]][head[1]] = game_const.GridType.OWN_HEAD

    # body 包含了 head 需要先赋值body
    grid[blue_body_list[0], blue_body_list[1]] = game_const.GridType.BLUE_BODY
    grid[blue_head_list[0], blue_head_list[1]] = game_const.GridType.BLUE_HEAD
    grid[purple_body_list[0], purple_body_list[1]] = game_const.GridType.PURPLE_BODY
    grid[purple_head_list[0], purple_head_list[1]] = game_const.GridType.PURPLE_HEAD
    grid[pink_body_list[0], pink_body_list[1]] = game_const.GridType.PINK_BODY
    grid[pink_head_list[0], pink_head_list[1]] = game_const.GridType.PINK_HEAD

    grid[yellow_crystal_list[0], yellow_crystal_list[1]] = game_const.GridType.YELLOW_CRYSTAL
    grid[green_speed_list[0], green_speed_list[1]] = game_const.GridType.GREEN_SPEED
    grid[blue_diamond_list[0], blue_diamond_list[1]] = game_const.GridType.BLUE_DIAMOND

    grid[bomb_list[0], bomb_list[1]] = game_const.GridType.BOMB

    # 是否可到达
    can_go_grid_list: list[NDArray[np.uint8]] = []
    for direction in range(4):
        can_go_grid_list.append(
            cal_can_go_grid(
                last_info=last_info,
                grid=grid,
                head=head,
                direction=direction,
                empty_list=empty_list,
                blue_head_list=blue_head_list,
                purple_head_list=purple_head_list,
                pink_head_list=pink_head_list,
                yellow_crystal_list=yellow_crystal_list,
                green_speed_list=green_speed_list,
                blue_diamond_list=blue_diamond_list,
            )
        )

    return head, grid, can_go_grid_list


def cal_can_go_grid(
        last_info: ZzzSnakeGameInfo,
        grid: NDArray[np.uint8],
        head: tuple[int, int],
        direction: int,
        empty_list: tuple[list[int], list[int]],
        blue_head_list: tuple[list[int], list[int]],
        purple_head_list: tuple[list[int], list[int]],
        pink_head_list: tuple[list[int], list[int]],
        yellow_crystal_list: tuple[list[int], list[int]],
        green_speed_list: tuple[list[int], list[int]],
        blue_diamond_list: tuple[list[int], list[int]],
) -> NDArray[np.uint8]:
    """
    下一步沿某个方向移动 计算该方向上可以到达的区域

    Args:
        grid: 当前网格
        head: 当前蛇头
        direction: 将要移动的方向
        last_info: 上一次分析结果

    Returns:
        can_go_grid: 可到达的区域 0=不可到达 1=未知 2=可到达
    """
    can_go_grid = np.full((game_const.GRID_ROWS, game_const.GRID_COLS), 0, dtype=np.int8)

    if last_info is not None and game_utils.is_opposite_direction(last_info.real_direction, direction):
        return can_go_grid

    if head is None:
        return can_go_grid

    # 判断下一个位置是否合法
    next_pos = game_utils.cal_next_position(head, direction)
    if not game_utils.is_pos_in_grid(next_pos):  # 出界了
        return can_go_grid

    next_grid_type = grid[next_pos[0], next_pos[1]]
    if not (next_grid_type == game_const.GridType.EMPTY
        or next_grid_type == game_const.GridType.YELLOW_CRYSTAL
        or next_grid_type == game_const.GridType.BLUE_DIAMOND
        or next_grid_type == game_const.GridType.GREEN_SPEED
    ):  # 不是可以移动的位置
        return can_go_grid

    can_go_grid[empty_list[0], empty_list[1]] = 1
    can_go_grid[yellow_crystal_list[0], yellow_crystal_list[1]] = 1
    can_go_grid[blue_diamond_list[0], blue_diamond_list[1]] = 1
    can_go_grid[green_speed_list[0], green_speed_list[1]] = 1
    can_go_grid[next_pos[0], next_pos[1]] = 2

    bfs_queue: list[tuple[int, int]] = []
    for y, x in zip(blue_head_list[0], blue_head_list[1]):
        bfs_queue.append((int(y), int(x)))
    for y, x in zip(purple_head_list[0], purple_head_list[1]):
        bfs_queue.append((int(y), int(x)))
    for y, x in zip(pink_head_list[0], pink_head_list[1]):
        bfs_queue.append((int(y), int(x)))
    bfs_queue.append(next_pos)  # 最后放自己的蛇头 因为这个位置是已经移动了一步的 让敌方蛇先移动
    while len(bfs_queue) > 0:
        cur_pos = bfs_queue.pop(0)
        for direction in range(4):
            next_pos = game_utils.cal_next_position(cur_pos, direction)
            if not game_utils.is_pos_in_grid(next_pos):  # 出界
                continue
            if can_go_grid[next_pos[0]][next_pos[1]] != 1:  # 已被占领
                continue

            can_go_grid[next_pos[0], next_pos[1]] = can_go_grid[cur_pos[0], cur_pos[1]]
            bfs_queue.append(next_pos)

    return can_go_grid


class GridDigitFormat:

    def __init__(self, color: tuple[int, int, int], font_scale: float, thickness: int):
        self.color: tuple[int, int, int] = color
        self.font_scale: float = font_scale
        self.thickness: int = thickness

def grid_to_image(grid: NDArray[np.uint8]) -> MatLike:
    """
    将二维数组转换为网格图像

    Args:
        grid: 二维数组/列表

    Returns:
        OpenCV图像
    """
    # 将输入转换为numpy数组
    data = np.array(grid)
    rows, cols = data.shape

    # 计算图像尺寸
    img_width = game_const.GAME_RECT[2] - game_const.GAME_RECT[0]
    img_height = game_const.GAME_RECT[3] - game_const.GAME_RECT[1]

    row_height = img_height * 1.0 / game_const.GRID_ROWS
    col_width = img_width * 1.0 / game_const.GRID_COLS

    # 创建空白白色图像
    img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255

    # 设置字体
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 绘制参数 需要与 game_const.GridType 对应
    digit_format: list[GridDigitFormat] = [None for _ in game_const.GridType]
    digit_format[game_const.GridType.UNKNOWN] = GridDigitFormat((0, 0, 0), 0.8, 2)
    digit_format[game_const.GridType.EMPTY] = GridDigitFormat((255, 255, 255), 0.8, 2)
    digit_format[game_const.GridType.OWN_HEAD] = GridDigitFormat((255, 73, 42), 0.8, 3)
    digit_format[game_const.GridType.OWN_BODY] = GridDigitFormat((255, 73, 42), 0.6, 2)
    digit_format[game_const.GridType.BLUE_HEAD] = GridDigitFormat((112, 71, 255), 0.8, 3)
    digit_format[game_const.GridType.BLUE_BODY] = GridDigitFormat((112, 71, 255), 0.6, 2)
    digit_format[game_const.GridType.PURPLE_HEAD] = GridDigitFormat((255, 35, 255), 0.8, 3)
    digit_format[game_const.GridType.PURPLE_BODY] = GridDigitFormat((255, 35, 255), 0.6, 2)
    digit_format[game_const.GridType.PINK_HEAD] = GridDigitFormat((255, 0, 0), 0.8, 3)
    digit_format[game_const.GridType.PINK_BODY] = GridDigitFormat((255, 0, 0), 0.6, 2)
    digit_format[game_const.GridType.YELLOW_CRYSTAL] = GridDigitFormat((255, 189, 68), 0.8, 3)
    digit_format[game_const.GridType.GREEN_SPEED] = GridDigitFormat((96, 255, 168), 0.8, 3)
    digit_format[game_const.GridType.BLUE_DIAMOND] = GridDigitFormat((55, 231, 205), 0.8, 3)
    digit_format[game_const.GridType.BOMB] = GridDigitFormat((193, 79, 79), 0.8, 3)

    # 绘制网格线和数字
    for i in range(rows + 1):
        # 画水平线
        y = int(i * row_height)
        cv2.line(img, (0, y), (img_width, y), (0, 0, 0), 2)

        if i < rows:
            for j in range(cols + 1):
                # 画垂直线
                x = int(j * col_width)
                cv2.line(img, (x, 0), (x, img_height), (0, 0, 0), 2)

                if j < cols:
                    # 在单元格中心绘制数字
                    text = str(data[i, j])
                    format: GridDigitFormat = digit_format[int(text)]
                    text_size = cv2.getTextSize(text, font, format.font_scale, format.thickness)[0]
                    text_x = int(x + (col_width - text_size[0]) / 2.0)
                    text_y = int(y + (row_height + text_size[1]) / 2.0)
                    cv2.putText(img, text, (text_x, text_y), font, format.font_scale, format.color, format.thickness)

    return img


def __debug_analyse():
    original_height: int = zzz_snake_gym.game_const.GAME_RECT[3] - zzz_snake_gym.game_const.GAME_RECT[1]
    original_width = zzz_snake_gym.game_const.GAME_RECT[2] - zzz_snake_gym.game_const.GAME_RECT[0]
    scale = 1
    analyzer = ZzzSnakeAnalyzer((original_width // scale, original_height // scale))
    from zzz_snake_gym import cv2_utils
    from zzz_snake_gym import debug_utils
    screenshot = debug_utils.get_debug_image('_1745146637092')
    import time
    start_time = time.time()
    for _ in range(100):
        result = analyzer.analyse(screenshot, time.time(), None)
    print(time.time() - start_time)

    cv2_utils.show_image(result.game_part, win_name='game_part')

    cv2_utils.show_image(grid_to_image(result.grid), win_name='grid_img')
    for i in range(4):
        cv2_utils.show_image(grid_to_image(result.can_go_grid_list[i]), win_name=f'can_go_grid_{i}_img')
    print(result.can_go_cnt)
    print(result.can_go_reward_cnt)
    print(result.direction_danger)
    cv2.waitKey(0)


if __name__ == '__main__':
    __debug_analyse()