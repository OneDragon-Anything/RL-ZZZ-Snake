from enum import IntEnum
from typing import Optional, ClassVar

import cv2
import numpy as np
from cv2.typing import MatLike
from numpy._typing import NDArray

import zzz_snake_gym.game_const
from zzz_snake_gym import game_utils, game_const, thread_utils, cv2_utils
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
            predict_head_pos: tuple[int, int],
            head_move_direction: int,
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

        self.predict_head_pos: tuple[int, int] = predict_head_pos  # 当前画面和上一帧结合计算的蛇头位置
        self.head_move_direction: int = head_move_direction  # 当前画面和上一帧画面 使用识别的蛇头位置 计算出来的正在移动的方向
        self.grid: NDArray[np.uint8] = grid
        self.can_go_grid_list: list[NDArray[np.uint8]] = can_go_grid_list

        self.can_go_cnt: list[int] = []  # 各方向可前往的格子数量
        self.can_go_reward_cnt: list[int] = []  # 各方向可前往的奖励格子数量
        self.direction_danger: list[int] = []  # 各方向的下一步是否危险
        self.direction_reward: list[int] = []  # 各方向的下一步是否奖励
        self.total_reward_cnt: int = max(total_reward_cnt, 1)
        self.cal_by_direction()

        self.press_direction: int = -1  # 按键的方向 初始化还没有按键
        self.effective_direction: int = -1  # 应该生效的方向

        self.real_direction: int = -1  # 真实的方向 即当前动作可能不生效 需要保持上一个方向
        self.predict_next_head_pos: Optional[tuple[int, int]] = None  # 根据真实的方向 预估下一个位置

        self.dis_to_reward: int = ZzzSnakeGameInfo.INF_DIS
        self.left_top_reward_dx: int = -game_const.GRID_COLS
        self.left_top_reward_dy: int = -game_const.GRID_ROWS
        self.left_bottom_reward_dx: int = -game_const.GRID_COLS
        self.left_bottom_reward_dy: int = game_const.GRID_ROWS
        self.right_top_reward_dx: int = game_const.GRID_COLS
        self.right_top_reward_dy: int = -game_const.GRID_ROWS
        self.right_bottom_reward_dx: int = game_const.GRID_COLS
        self.right_bottom_reward_dy: int = game_const.GRID_ROWS

        self.cal_reward_dis()

        self.current_in_boundary: bool = False  # 当前位置是否在边界上
        self.is_away_boundary: bool = False  # 当前动作是否让蛇远离边界
        self.dis_to_boundary: int = ZzzSnakeGameInfo.INF_DIS  # 与边界的距离
        self.cal_boundary_dis()  # 计算边界相关的值

        # 根据坐标进行的一系列判断
        self.is_next_danger: bool = False  # 预估下一步是否危险 边界/障碍/炸弹/可移动格子最少
        self.is_next_reward: bool = False  # 预估下一步到达食物

    def set_direction(self, press_direction: int, last_info: Optional['ZzzSnakeGameInfo']) -> None:
        """
        设置本次执行的动作 同时预测下一个位置
        Args:
            press_direction: 本次按键的动作
            last_info: 上一个游戏信息

        Returns:

        """
        # 非法的方向
        if press_direction < 0 or press_direction > 3:
            return

        self.press_direction = press_direction
        self.effective_direction = press_direction

        if self.head_move_direction != -1:
            if (last_info is not None
                    and self.head_move_direction != last_info.press_direction
                and not game_utils.is_opposite_direction(self.head_move_direction, last_info.press_direction)
            ):
                # 上一个动作还没有生效的话 继续做上一次的动作
                self.effective_direction = last_info.press_direction

        # 如果生效方向和蛇头移动方向相反 则设置为无效
        if game_utils.is_opposite_direction(self.head_move_direction, self.effective_direction):
            self.effective_direction = self.head_move_direction

        # 计算下一个位置 和其它标记位信息
        if self.predict_head_pos is not None:
            self.predict_next_head_pos = game_utils.cal_next_position(self.predict_head_pos, self.effective_direction)
            self.is_next_danger = self.cal_is_next_danger()
            self.is_next_reward = self.cal_is_next_reward()

    def cal_boundary_dis(self) -> None:
        """
        计算边界相关的距离
        Returns:
            None
        """
        if self.predict_head_pos is None:
            return

        dis = [
            self.predict_head_pos[0],
            zzz_snake_gym.game_const.GRID_ROWS - 1 - self.predict_head_pos[0],
            self.predict_head_pos[1],
            zzz_snake_gym.game_const.GRID_COLS - 1 - self.predict_head_pos[1],
        ]
        if 0 <= self.head_move_direction < 4:
            self.dis_to_boundary = dis[self.head_move_direction]
        else:
            self.dis_to_boundary = min(dis)

        self.is_away_boundary = False
        if self.predict_head_pos[0] == 0 or self.predict_head_pos[0] == zzz_snake_gym.game_const.GRID_ROWS - 1:
            self.current_in_boundary = True
            if not(self.predict_head_pos[1] == 0 or self.predict_head_pos[1] == zzz_snake_gym.game_const.GRID_COLS - 1):
                if self.predict_head_pos[0] < 2 and self.effective_direction == 1:
                    self.is_away_boundary = True
                elif self.predict_head_pos[0] > game_const.GRID_ROWS - 3 and self.effective_direction == 0:
                    self.is_away_boundary = True
        elif self.predict_head_pos[1] == 0 or self.predict_head_pos[1] == zzz_snake_gym.game_const.GRID_COLS - 1:
            self.current_in_boundary = True
            if not (self.predict_head_pos[0] == 0 or self.predict_head_pos[0] == zzz_snake_gym.game_const.GRID_ROWS - 1):
                if self.predict_head_pos[1] < 2 and self.effective_direction == 3:
                    self.is_away_boundary = True
                elif self.predict_head_pos[1] > game_const.GRID_COLS - 3 and self.effective_direction == 2:
                    self.is_away_boundary = True
        else:
            self.current_in_boundary = False



    def cal_is_next_danger(self) -> bool:
        """
        Returns:
            当前动作是否朝未知格子前进
        """
        # 预估位置非法
        if not game_utils.is_pos_in_grid(self.predict_next_head_pos):
            return True

        if not 0 <= self.effective_direction < len(self.direction_danger):
            return False

        # 下一步就撞墙
        if self.direction_danger[self.effective_direction] == 1:
            return True

        # 该方向能走的格子最少
        min_can_go_cnt: int = 999
        min_can_go_directions: list[int] = []
        for direction in range(4):
            if self.can_go_cnt[direction] < min_can_go_cnt:
                min_can_go_cnt = self.can_go_cnt[direction]
                min_can_go_directions.clear()
                min_can_go_directions.append(direction)
            elif self.can_go_cnt[direction]== min_can_go_cnt:
                min_can_go_directions.append(direction)

        if self.effective_direction in min_can_go_directions:
            return True

        return False

    def cal_reward_dis(self) -> None:
        """
        计算最近的奖励点距离及其偏移量
        """
        if self.predict_head_pos is None:
            return

        # 创建奖励点的布尔掩码
        reward_mask = ((self.grid == game_const.GridType.YELLOW_CRYSTAL) |
                       (self.grid == game_const.GridType.GREEN_SPEED) |
                       (self.grid == game_const.GridType.BLUE_DIAMOND) |
                       (self.grid == game_const.GridType.GOLD_STAR)
                       )

        # 获取所有奖励点的坐标
        reward_positions = np.argwhere(reward_mask)

        if len(reward_positions) == 0:
            return

        # 计算所有奖励点到头部的偏移量
        offsets = reward_positions - np.array(self.predict_head_pos)

        # 计算曼哈顿距离
        distances = np.abs(offsets).sum(axis=1)

        # 找到最小距离的索引
        min_idx = np.argmin(distances)

        # 获取最小距离和对应的偏移量
        min_distance = distances[min_idx]
        dy, dx = offsets[min_idx]
        self.dis_to_reward = min_distance

        # 找出在左上方的奖励位置 (dy <= 0 且 dx <= 0)
        left_top_idx = np.where((offsets[:, 0] <= 0) & (offsets[:, 1] <= 0))[0]
        if len(left_top_idx) > 0:
            self.left_top_reward_dy = int(offsets[left_top_idx[0], 0])
            self.left_top_reward_dx = int(offsets[left_top_idx[0], 1])

        # 找出在左下方的奖励位置 (dy >= 0 且 dx <= 0)
        left_bottom_idx = np.where((offsets[:, 0] >= 0) & (offsets[:, 1] <= 0))[0]
        if len(left_bottom_idx) > 0:
            self.left_bottom_reward_dy = int(offsets[left_bottom_idx[0], 0])
            self.left_bottom_reward_dx = int(offsets[left_bottom_idx[0], 1])

        # 找出在右上方的奖励位置 (dy <= 0 且 dx >= 0)
        right_top_idx = np.where((offsets[:, 0] <= 0) & (offsets[:, 1] >= 0))[0]
        if len(right_top_idx) > 0:
            self.right_top_reward_dy = int(offsets[right_top_idx[0], 0])
            self.right_top_reward_dx = int(offsets[right_top_idx[0], 1])

        # 找出在右下方的奖励位置 (dy >= 0 且 dx >= 0)
        right_bottom_idx = np.where((offsets[:, 0] >= 0) & (offsets[:, 1] >= 0))[0]
        if len(right_bottom_idx) > 0:
            self.right_bottom_reward_dy = int(offsets[right_bottom_idx[0], 0])
            self.right_bottom_reward_dx = int(offsets[right_bottom_idx[0], 1])

    def cal_by_direction(self) -> None:
        """
        计算4个方向将会遇到的情况
        """
        for direction in range(4):
            if self.predict_head_pos is None:
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
                | (self.grid == game_const.GridType.GOLD_STAR)
            )

            can_go_y, can_go_x = np.where(can_go_idx)
            reward_y, reward_x = np.where(can_go_idx & reward_idx)

            self.can_go_cnt.append(len(can_go_x))
            self.can_go_reward_cnt.append(len(reward_x))

            next_pos = game_utils.cal_next_position(self.predict_head_pos, direction)
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
                    or next_grid_type == game_const.GridType.GOLD_STAR
                ):  # 奖励格子
                    self.direction_danger.append(0)
                    self.direction_reward.append(1)
                else:  # 剩余都是不可以移动的位置
                    self.direction_danger.append(1)
                    self.direction_reward.append(0)

    def cal_is_next_reward(self) -> bool:
        return self.effective_direction is not None and self.press_direction == self.effective_direction and self.direction_reward[self.effective_direction]

    def is_grid_reward(self, head_pos: tuple[int, int]) -> bool:
        """
        目标格子是否奖励
        Args:
            head_pos: 目标格子

        Returns:
            is_reward: 是否奖励
        """
        return game_utils.is_pos_in_grid(head_pos) and self.grid[head_pos[0], head_pos[1]] in [
            game_const.GridType.YELLOW_CRYSTAL,
            game_const.GridType.BLUE_DIAMOND,
            game_const.GridType.GREEN_SPEED,
            game_const.GridType.GOLD_STAR,
        ]


class ZzzSnakeAnalyzer:

    def __init__(self, target_size: tuple[int, int]):
        self.target_size = target_size  # 最后输出的尺寸 高、宽

    def analyse(
            self,
            screenshot: MatLike,
            current_time: float,
            last_info: ZzzSnakeGameInfo,
            should_be_move: bool
    ) -> ZzzSnakeGameInfo:
        """
        对当前画面进行分析 并更新上一帧的分析结果

        平均18ms
        Args:
            screenshot: 整个游戏画面
            current_time: 截图时间
            last_info: 上一次的分析结果
            should_be_move: 根据时间 判断是否应该发生了移动

        Returns:
            current_info: 当前帧的分析结果
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
        # own_head_future = thread_utils.submit(cv2.inRange, hsv_game_part, game_const.OWN_HEAD_HSV_RANGE[0], game_const.OWN_HEAD_HSV_RANGE[1])
        own_head_eye_future = thread_utils.submit(cv2.inRange, hsv_game_part, game_const.OWN_HEAD_EYE_HSV_RANGE[0], game_const.OWN_HEAD_EYE_HSV_RANGE[1])
        own_body_future = thread_utils.submit(cv2.inRange, hsv_game_part, game_const.OWN_BODY_HSV_RANGE[0], game_const.OWN_BODY_HSV_RANGE[1])
        blue_head_future = thread_utils.submit(cv2.inRange, hsv_game_part, game_const.BLUE_HEAD_HSV_RANGE[0], game_const.BLUE_HEAD_HSV_RANGE[1])
        blue_body_future = thread_utils.submit(cv2.inRange, hsv_game_part, game_const.BLUE_BODY_HSV_RANGE[0], game_const.BLUE_BODY_HSV_RANGE[1])
        purple_head_future = thread_utils.submit(cv2.inRange, hsv_game_part, game_const.PURPLE_HEAD_HSV_RANGE[0], game_const.PURPLE_HEAD_HSV_RANGE[1])
        purple_body_future = thread_utils.submit(cv2.inRange, hsv_game_part, game_const.PURPLE_BODY_HSV_RANGE[0], game_const.PURPLE_BODY_HSV_RANGE[1])
        pink_head_future = thread_utils.submit(cv2.inRange, hsv_game_part, game_const.PINK_HEAD_HSV_RANGE[0], game_const.PINK_HEAD_HSV_RANGE[1])
        pink_head_future_2 = thread_utils.submit(cv2.inRange, hsv_game_part, game_const.PINK_HEAD_HSV_RANGE_2[0], game_const.PINK_HEAD_HSV_RANGE_2[1])
        pink_body_future = thread_utils.submit(cv2.inRange, hsv_game_part, game_const.PINK_BODY_HSV_RANGE[0], game_const.PINK_BODY_HSV_RANGE[1])
        gold_head_future = thread_utils.submit(cv2.inRange, hsv_game_part, game_const.GOLD_HEAD_HSV_RANGE[0], game_const.GOLD_HEAD_HSV_RANGE[1])
        gold_body_future = thread_utils.submit(cv2.inRange, hsv_game_part, game_const.GOLD_BODY_HSV_RANGE[0], game_const.GOLD_BODY_HSV_RANGE[1])
        yellow_crystal_future = thread_utils.submit(cv2.inRange, hsv_game_part, game_const.YELLOW_CRYSTAL_HSV_RANGE[0], game_const.YELLOW_CRYSTAL_HSV_RANGE[1])
        green_speed_future = thread_utils.submit(cv2.inRange, hsv_game_part, game_const.GREEN_SPEED_HSV_RANGE[0], game_const.GREEN_SPEED_HSV_RANGE[1])
        blue_diamond_future = thread_utils.submit(cv2.inRange, hsv_game_part, game_const.BLUE_DIAMOND_HSV_RANGE[0], game_const.BLUE_DIAMOND_HSV_RANGE[1])
        gold_star_future = thread_utils.submit(cv2.inRange, hsv_game_part, game_const.GOLD_STAR_HSV_RANGE[0], game_const.GOLD_STAR_HSV_RANGE[1])
        bomb_future = thread_utils.submit(cv2.inRange, hsv_game_part, game_const.BOMB_HSV_RANGE[0], game_const.BOMB_HSV_RANGE[1])
        bomb_future_2 = thread_utils.submit(cv2.inRange, hsv_game_part, game_const.BOMB_HSV_RANGE_2[0], game_const.BOMB_HSV_RANGE_2[1])
        grey_stone_future = thread_utils.submit(cv2.inRange, hsv_game_part, game_const.GREY_STONE_HSV_RANGE[0], game_const.GREY_STONE_HSV_RANGE[1])

        empty_mask = empty_future.result()
        # own_head_mask = own_head_future.result()
        own_head_eye_mask = own_head_eye_future.result()
        own_body_mask = own_body_future.result()
        blue_head_mask = blue_head_future.result()
        blue_body_mask = blue_body_future.result()
        purple_head_mask = purple_head_future.result()
        purple_body_mask = purple_body_future.result()
        pink_head_mask = pink_head_future.result()
        pink_head_mask_2 = pink_head_future_2.result()
        pink_head_mask = cv2.bitwise_or(pink_head_mask, pink_head_mask_2)
        pink_body_mask = pink_body_future.result()
        gold_head_mask = gold_head_future.result()
        gold_body_mask = gold_body_future.result()
        yellow_crystal_mask = yellow_crystal_future.result()
        green_speed_mask = green_speed_future.result()
        blue_diamond_mask = blue_diamond_future.result()
        gold_star_mask = gold_star_future.result()
        bomb_mask = bomb_future.result()
        bomb_mask_2 = bomb_future_2.result()
        bomb_mask = cv2.bitwise_or(bomb_mask, bomb_mask_2)
        grey_stone_mask = grey_stone_future.result()

        empty_future = thread_utils.submit(get_coordinate_by_mask_center, empty_mask, 10, 'max', 125)
        # own_head_future = thread_utils.submit(get_coordinate_by_mask_center, own_head_mask, 1, 'avg', 255)
        own_head_eye_future = thread_utils.submit(get_grid_coordinate_by_mask, own_head_eye_mask)
        own_body_future = thread_utils.submit(get_coordinate_by_mask_center, own_body_mask, 10, 'max', 255)
        blue_head_future = thread_utils.submit(get_coordinate_by_mask_center, blue_head_mask, 10, 'max', 255)
        blue_body_future = thread_utils.submit(get_coordinate_by_mask_center, blue_body_mask, 10, 'max', 255)
        purple_head_future = thread_utils.submit(get_coordinate_by_mask_center, purple_head_mask, 10, 'max', 255)
        purple_body_future = thread_utils.submit(get_coordinate_by_mask_center, purple_body_mask, 10, 'max', 255)
        pink_head_future = thread_utils.submit(get_coordinate_by_mask_center, pink_head_mask, 10, 'max', 255)
        pink_body_future = thread_utils.submit(get_coordinate_by_mask_center, pink_body_mask, 10, 'max', 255)
        gold_head_future = thread_utils.submit(get_coordinate_by_mask_center, gold_head_mask, 10, 'max', 255)
        gold_body_future = thread_utils.submit(get_coordinate_by_mask_center, gold_body_mask, 10, 'max', 255)

        yellow_crystal_future = thread_utils.submit(get_coordinate_by_mask_center, yellow_crystal_mask, 2, 'avg', 125)
        green_speed_future = thread_utils.submit(get_coordinate_by_mask_center, green_speed_mask, 2, 'avg', 125)
        blue_diamond_future = thread_utils.submit(get_coordinate_by_mask_center, blue_diamond_mask, 2, 'avg', 125)
        gold_star_future = thread_utils.submit(get_coordinate_by_mask_center, gold_star_mask, 2, 'avg', 125)
        bomb_future = thread_utils.submit(get_coordinate_by_mask_center, bomb_mask, 5, 'avg', 50)
        grey_stone_future = thread_utils.submit(get_coordinate_by_mask_center, grey_stone_mask, 2, 'avg', 125)

        empty_list = empty_future.result()
        # own_head_list = own_head_future.result()
        own_head_eye_list = own_head_eye_future.result()
        own_body_list = own_body_future.result()
        blue_head_list = blue_head_future.result()
        blue_body_list = blue_body_future.result()
        purple_head_list = purple_head_future.result()
        purple_body_list = purple_body_future.result()
        pink_head_list = pink_head_future.result()
        pink_body_list = pink_body_future.result()
        gold_head_list = gold_head_future.result()
        gold_body_list = gold_body_future.result()
        yellow_crystal_list = yellow_crystal_future.result()
        green_speed_list = green_speed_future.result()
        blue_diamond_list = blue_diamond_future.result()
        gold_star_list = gold_star_future.result()
        bomb_list = bomb_future.result()
        grey_stone_list = grey_stone_future.result()

        total_reward_cnt: int = len(yellow_crystal_list[0]) + len(green_speed_list[0]) + len(blue_diamond_list[0]) + len(gold_star_list[0])

        # 初始化一个包含空格的网格
        grid = np.full((game_const.GRID_ROWS, game_const.GRID_COLS), game_const.GridType.UNKNOWN, dtype=np.int8)
        grid[empty_list[0], empty_list[1]] = game_const.GridType.EMPTY

        # 使用空格预测头部
        hsv_head_pos = get_head_pos(own_head_eye_list, grid)
        head_move_direction = get_head_move_direction(hsv_head_pos, last_info)
        predict_head_pos = get_predict_head_pos(hsv_head_pos, last_info, should_be_move=should_be_move)

        # 为了兼容识别错误情况 先设置奖励 再设置其他障碍 保证一定不会丢失障碍信息 生存优先
        grid[yellow_crystal_list[0], yellow_crystal_list[1]] = game_const.GridType.YELLOW_CRYSTAL
        grid[green_speed_list[0], green_speed_list[1]] = game_const.GridType.GREEN_SPEED
        grid[blue_diamond_list[0], blue_diamond_list[1]] = game_const.GridType.BLUE_DIAMOND
        grid[gold_star_list[0], gold_star_list[1]] = game_const.GridType.GOLD_STAR

        grid[own_body_list[0], own_body_list[1]] = game_const.GridType.OWN_BODY

        # body 包含了 head 需要先赋值body
        grid[blue_body_list[0], blue_body_list[1]] = game_const.GridType.BLUE_BODY
        grid[blue_head_list[0], blue_head_list[1]] = game_const.GridType.BLUE_HEAD
        grid[purple_body_list[0], purple_body_list[1]] = game_const.GridType.PURPLE_BODY
        grid[purple_head_list[0], purple_head_list[1]] = game_const.GridType.PURPLE_HEAD
        grid[pink_body_list[0], pink_body_list[1]] = game_const.GridType.PINK_BODY
        grid[pink_head_list[0], pink_head_list[1]] = game_const.GridType.PINK_HEAD
        grid[gold_head_list[0], gold_head_list[1]] = game_const.GridType.GOLD_HEAD
        grid[gold_body_list[0], gold_body_list[1]] = game_const.GridType.GOLD_BODY

        grid[bomb_list[0], bomb_list[1]] = game_const.GridType.BOMB
        grid[grey_stone_list[0], grey_stone_list[1]] = game_const.GridType.GREY_STONE

        # 由于吃了无敌后会变色 会被误判成其他蛇 因此自己的蛇头需要最后在赋值
        if predict_head_pos is not None:
            grid[predict_head_pos[0]][predict_head_pos[1]] = game_const.GridType.OWN_HEAD

        # 是否可到达
        can_go_grid_list: list[NDArray[np.uint8]] = []
        for direction in range(4):
            can_go_grid_list.append(
                cal_can_go_grid(
                    last_info=last_info,
                    grid=grid,
                    head_pos=predict_head_pos,
                    direction=direction,
                    empty_list=empty_list,
                    blue_head_list=blue_head_list,
                    purple_head_list=purple_head_list,
                    pink_head_list=pink_head_list,
                    gold_head_list=gold_head_list,
                    yellow_crystal_list=yellow_crystal_list,
                    green_speed_list=green_speed_list,
                    blue_diamond_list=blue_diamond_list,
                    gold_star_list=gold_star_list,
                )
            )

        info = ZzzSnakeGameInfo(
            start_time=current_time if last_info is None else last_info.start_time,
            current_time=current_time,
            screenshot=screenshot,
            game_part=game_part,
            hsv_game_part=hsv_game_part,
            score=score,
            game_over=game_over,
            predict_head_pos=predict_head_pos,
            head_move_direction=head_move_direction,
            grid=grid,
            can_go_grid_list=can_go_grid_list,
            total_reward_cnt=total_reward_cnt,
        )

        return info
    

def get_grid_coordinate_by_mask(
        mask: MatLike,
) -> tuple[NDArray, NDArray]:
    """
    根据掩码图 获取对应在网格上的坐标
    Args:
        mask: 掩码图

    Returns:
        y_list, x_list: 网格坐标
    """
    h, w = mask.shape

    # 计算每个格子的高度和宽度
    cell_h = h * 1.0 / game_const.GRID_ROWS
    cell_w = w * 1.0 / game_const.GRID_COLS

    # 遍历整个mask，找出所有值为255的像素点
    y_coords, x_coords = np.where(mask == 255)
    y_coords = y_coords // cell_h
    y_coords = y_coords.astype(dtype=np.uint8)
    x_coords = x_coords // cell_w
    x_coords = x_coords.astype(dtype=np.uint8)

    # 将集合转换为列表并返回
    return y_coords, x_coords


def get_coordinate_by_mask_center(
        mask: MatLike,
        radius: int = 1,
        sample_way: str = 'avg',
        lower: int = 255
) -> tuple[NDArray, NDArray]:
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


def get_head_pos(
        own_head_list: tuple[list[int], list[int]],
        grid: NDArray[np.uint8]
) -> tuple[int, int]:
    """
    根据识别到的头部坐标列表 选出现真正的头部坐标
    如果头部有多个坐标 选择包含空格区域的且出现最多的坐标
    Args:
        own_head_list: 识别到的头部坐标列表
        grid: 网格信息 不包含自己的头部

    Returns:
        head_pos: 头部坐标
    """
    head_pos_map: dict[tuple[int, int], int] = {}
    with_empty_grid: set[tuple[int, int]] = set()
    for y, x in zip(own_head_list[0], own_head_list[1]):
        pos = (int(y), int(x))
        head_pos_cnt = head_pos_map.get(pos, 0) + 1
        head_pos_map[pos] = head_pos_cnt
        if grid[pos[0], pos[1]] == game_const.GridType.EMPTY:
            with_empty_grid.add(pos)

    head = None
    head_pos_max_cnt: int = 0
    for pos, cnt in head_pos_map.items():
        if len(with_empty_grid) > 0 and pos not in with_empty_grid:
            continue
        if cnt > head_pos_max_cnt:
            head = pos
            head_pos_max_cnt = cnt

    return head


def get_predict_head_pos(
        head_pos: tuple[int, int],
        last_info: ZzzSnakeGameInfo,
        should_be_move: bool,
) -> Optional[tuple[int, int]]:
    """
    结合当前画面识别结果和上一帧信息 计算当前的坐标
    Args:
        head_pos: 当前的识别到的头部
        last_info: 上一个位置信息
        should_be_move: 是否应该移动

    Returns:
        predict_head_pos: 计算当前的坐标
    """
    if head_pos is not None:
        return head_pos

    # 使用颜色匹配不到的时候 使用上一次预测的位置
    if head_pos is None and last_info is not None and should_be_move:
        if last_info.predict_next_head_pos is not None and not last_info.is_next_danger:
            return last_info.predict_next_head_pos
        elif last_info.predict_head_pos is not None:
            return last_info.predict_head_pos

    return None


def get_head_move_direction(
        head_pos: tuple[int, int],
        last_info: ZzzSnakeGameInfo,
) -> int:
    """
    根据头部坐标的变动 判断当前的移动方向

    Args:
        head_pos: 当前的识别到的头部
        last_info: 上一个位置信息

    Returns:
        direction: 移动方向
    """
    if head_pos is None or last_info is None or last_info.predict_head_pos is None:
        return -1

    if head_pos[0] == last_info.predict_head_pos[0]:
        if head_pos[1] > last_info.predict_head_pos[1]:
            return 3
        elif head_pos[1] < last_info.predict_head_pos[1]:
            return 2
    elif head_pos[1] == last_info.predict_head_pos[1]:
        if head_pos[0] > last_info.predict_head_pos[0]:
            return 1
        elif head_pos[0] < last_info.predict_head_pos[0]:
            return 0

    return -1


def cal_can_go_grid(
        last_info: ZzzSnakeGameInfo,
        grid: NDArray[np.uint8],
        head_pos: tuple[int, int],
        direction: int,
        empty_list: tuple[list[int], list[int]],
        blue_head_list: tuple[list[int], list[int]],
        purple_head_list: tuple[list[int], list[int]],
        pink_head_list: tuple[list[int], list[int]],
        gold_head_list: tuple[list[int], list[int]],
        yellow_crystal_list: tuple[list[int], list[int]],
        green_speed_list: tuple[list[int], list[int]],
        gold_star_list: tuple[list[int], list[int]],
        blue_diamond_list: tuple[list[int], list[int]],
) -> NDArray[np.uint8]:
    """
    下一步沿某个方向移动 计算该方向上可以到达的区域

    Args:
        grid: 当前网格
        head_pos: 当前蛇头
        direction: 将要移动的方向
        last_info: 上一次分析结果

    Returns:
        can_go_grid: 可到达的区域 0=不可到达 1=未知 2=可到达
    """
    # 初始化成所有都不可到达
    can_go_grid = np.full((game_const.GRID_ROWS, game_const.GRID_COLS), 0, dtype=np.int8)

    if last_info is not None and game_utils.is_opposite_direction(last_info.real_direction, direction):
        return can_go_grid

    if head_pos is None:
        return can_go_grid

    # 判断下一个位置是否合法
    next_pos = game_utils.cal_next_position(head_pos, direction)
    if not game_utils.is_pos_in_grid(next_pos):  # 出界了
        return can_go_grid

    next_grid_type = grid[next_pos[0], next_pos[1]]
    if not (next_grid_type == game_const.GridType.EMPTY
            or next_grid_type == game_const.GridType.YELLOW_CRYSTAL
            or next_grid_type == game_const.GridType.BLUE_DIAMOND
            or next_grid_type == game_const.GridType.GREEN_SPEED
            or next_grid_type == game_const.GridType.GOLD_STAR
    ):  # 不是可以移动的位置
        return can_go_grid

    # 空点和奖励点设置为未知 需要搜索才知道能不能到达
    can_go_grid[empty_list[0], empty_list[1]] = 1
    can_go_grid[yellow_crystal_list[0], yellow_crystal_list[1]] = 1
    can_go_grid[blue_diamond_list[0], blue_diamond_list[1]] = 1
    can_go_grid[green_speed_list[0], green_speed_list[1]] = 1
    can_go_grid[gold_star_list[0], gold_star_list[1]] = 1

    # 自己的蛇头设置为可到达
    can_go_grid[next_pos[0], next_pos[1]] = 2

    # 遍历各个蛇头 加入搜索队列
    bfs_queue: list[tuple[int, int]] = []
    for y, x in zip(blue_head_list[0], blue_head_list[1]):
        bfs_queue.append((int(y), int(x)))
    for y, x in zip(purple_head_list[0], purple_head_list[1]):
        bfs_queue.append((int(y), int(x)))
    for y, x in zip(pink_head_list[0], pink_head_list[1]):
        bfs_queue.append((int(y), int(x)))
    for y, x in zip(gold_head_list[0], gold_head_list[1]):
        bfs_queue.append((int(y), int(x)))
    bfs_queue.append(next_pos)  # 最后放自己的蛇头 因为这个位置是已经移动了一步的 让敌方蛇先移动
    while len(bfs_queue) > 0:
        cur_pos = bfs_queue.pop(0)
        for direction in range(4):
            next_pos = game_utils.cal_next_position(cur_pos, direction)
            if not game_utils.is_pos_in_grid(next_pos):  # 出界
                continue
            if can_go_grid[cur_pos[0]][cur_pos[1]] == 1:  # 有可能识别错误了 蛇头和奖励重叠导致赋值成了1 这时候不进行后续搜索
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
    digit_format: list[Optional[GridDigitFormat]] = [None for _ in game_const.GridType]
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
    digit_format[game_const.GridType.GOLD_HEAD] = GridDigitFormat((255, 255, 0), 0.8, 3)
    digit_format[game_const.GridType.GOLD_BODY] = GridDigitFormat((255, 255, 0), 0.6, 2)
    digit_format[game_const.GridType.YELLOW_CRYSTAL] = GridDigitFormat((255, 189, 68), 0.8, 3)
    digit_format[game_const.GridType.GREEN_SPEED] = GridDigitFormat((95, 232, 150), 0.8, 3)
    digit_format[game_const.GridType.GOLD_STAR] = GridDigitFormat((255, 255, 0), 0.8, 3)
    digit_format[game_const.GridType.BLUE_DIAMOND] = GridDigitFormat((55, 231, 205), 0.8, 3)
    digit_format[game_const.GridType.BOMB] = GridDigitFormat((193, 79, 79), 0.8, 3)
    digit_format[game_const.GridType.GREY_STONE] = GridDigitFormat((108, 112, 138), 0.8, 3)

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
        result = analyzer.analyse(screenshot, time.time(), None, should_be_move=True)
    print(time.time() - start_time)

    cv2_utils.show_image(result.game_part, win_name='game_part')

    cv2_utils.show_image(grid_to_image(result.grid), win_name='grid_img')
    for i in range(4):
        cv2_utils.show_image(grid_to_image(result.can_go_grid_list[i]), win_name=f'can_go_grid_{i}_img')
    print(result.can_go_cnt)
    print(result.can_go_reward_cnt)
    print(result.direction_danger)
    cv2.waitKey(0)
    result.set_direction(0, result)



if __name__ == '__main__':
    __debug_analyse()