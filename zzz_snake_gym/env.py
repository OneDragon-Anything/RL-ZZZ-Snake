import os
import time
from typing import SupportsFloat, Any, Optional

import cv2
import gymnasium as gym
import numpy as np
from cv2.typing import MatLike
from gymnasium import spaces
from gymnasium.core import ObsType
from numpy._typing import NDArray

import zzz_snake_gym.game_const
from zzz_snake_gym import game_utils, cv2_utils, game_const, os_utils
from zzz_snake_gym.controller import ZzzSnakeController, KeyMouseZzzSnakeController
from zzz_snake_gym.performance_utils import timeit
from zzz_snake_gym.screen_capturer import ZzzSnakeScreenCapturer, DefaultZzzSnakeScreenCapturer
from zzz_snake_gym.snake_analyzer import ZzzSnakeAnalyzer, ZzzSnakeGameInfo


class ZzzSnakeEnv(gym.Env):

    def __init__(
            self,
            win_title: str = '绝区零',
            screen_capturer: ZzzSnakeScreenCapturer = None,
            controller: ZzzSnakeController = None,
            scale: int = 8,
            save_replay: bool = False,
            save_replay_dir: str = None,
    ):
        gym.Env.__init__(self)

        self.save_replay: bool = save_replay and save_replay_dir is not None  # 是否保存数据用于离线训练
        self.save_replay_dir: str = save_replay_dir  # 保存数据的目录
        self.save_idx: int = 0  # 保存数据的下标

        # 定义观察空间
        original_height: int = zzz_snake_gym.game_const.GAME_RECT[3] - zzz_snake_gym.game_const.GAME_RECT[1]
        original_width = zzz_snake_gym.game_const.GAME_RECT[2] - zzz_snake_gym.game_const.GAME_RECT[0]

        self.target_size = (
            original_width // scale,
            original_height // scale,
        )

        grid_types = len(game_const.GridType)
        self.observation_space: spaces.Dict = spaces.Dict({
            'image': spaces.Box(low=0, high=255, shape=(self.target_size[1], self.target_size[0], 3 + grid_types),
                                dtype=np.uint8),
            'grid': spaces.Box(low=0, high=1, shape=(game_const.GRID_ROWS, game_const.GRID_COLS, grid_types),
                               dtype=np.uint8),
            'feat_direction_cnt': spaces.Box(low=0, high=1, shape=(16,), dtype=np.float32),  # 朝4个方向移动 预估可遇到的各类型数量
            'feat_distance': spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32),  # 与各类型目标点的距离偏移量
            'last_action': spaces.Box(low=0, high=1, shape=(4,), dtype=np.uint8),
        })

        # 定义动作空间
        # 0=上，1=下，2=左，3=右
        self.action_space = spaces.Discrete(4)

        # 截图器 返回游戏画面的截图 RGB格式
        self.screen_capturer: ZzzSnakeScreenCapturer = DefaultZzzSnakeScreenCapturer(
            win_title=win_title) if screen_capturer is None else screen_capturer

        # 控制器
        self.controller: ZzzSnakeController = KeyMouseZzzSnakeController() if controller is None else controller

        # 分析器
        self.analyzer: ZzzSnakeAnalyzer = ZzzSnakeAnalyzer((original_height, original_width))

        # 上一帧的游戏信息
        self.last_info: Optional[ZzzSnakeGameInfo] = None
        # 上上一帧的游戏信息
        self.last_info_2: Optional[ZzzSnakeGameInfo] = None

    def get_init_game_info(
            self,
            current_time: float,
            screenshot: MatLike,
            last_info: Optional[ZzzSnakeGameInfo]
    ) -> ZzzSnakeGameInfo:
        """
        初始化游戏信息
        Returns:
            无
        """
        # 上一帧的游戏信息
        info = self.analyzer.analyse(screenshot, current_time, last_info)
        info.set_direction(0, last_info)
        return info

    def _get_reward(self, info: ZzzSnakeGameInfo):
        """

        Args:
            info: 当前的游戏信息

        Returns:
            奖励分值 在  [-1, 1] 范围
        """
        if info.game_over:  # 游戏结束
            return -1

        reward: float = 0
        # 分阶段
        if self.last_info.survival_seconds < 30:  # 前期只要存活就奖励
            reward += 0.05
        elif self.last_info.survival_seconds < 60:
            reward += 0.025
        elif self.last_info.survival_seconds < 90:
            reward += 0.0125
        else:   # 后期每步小惩罚 避免绕圈
            reward -= 0.01

        # 分数上升 奖励
        # if info.score > self.last_info.score:
        #     reward += info.score - self.last_info.score  # 使用分数差来做奖励

        # 本次操作非法
        if self.last_info.direction != self.last_info.real_direction:
            reward -= 0.5

        # 前往危险的格子要扣分 边界/障碍
        if self.last_info.is_predict_danger:
            # print('危险扣分')
            reward -= 1

        # 预估到达食物要加分
        if self.last_info.is_predict_reward:
            reward += 1
        elif (not self.last_info.is_predict_danger
              and self.last_info.dis_to_reward > info.dis_to_reward
        ):
            # 接近奖励要加分
            # print('朝向奖励加分',
            #       self.last_info.head, self.last_info.real_action,
            #       (self.last_info.reward_x_min, self.last_info.reward_x_max), (self.last_info.reward_y_min, self.last_info.reward_y_max))
            # 越靠近奖励 加分越多 至少加0.1分
            reward += max(0.5 - info.dis_to_reward * 0.1, 0.1)

        # 保证范围
        if reward < -1:
            reward = -1
        elif reward > 1:
            reward = 1

        return reward

    def _take_action(self, direction: int):
        self.controller.move(direction)
        self.last_info.set_direction(direction, self.last_info_2)

    def get_obs(self, info: ZzzSnakeGameInfo) -> dict:
        """
        大概6ms
        Args:
            info: 当前游戏信息

        Returns:
            处理后的观察空间
        """
        if info.game_over:
            return {
                'image': np.zeros(self.observation_space['image'].shape, dtype=np.uint8),
                'grid': np.zeros(self.observation_space['grid'].shape, dtype=np.uint8),
                'last_action': np.zeros(self.observation_space['last_action'].shape, dtype=np.uint8),
                'feat_direction_cnt': np.zeros(self.observation_space['feat_direction_cnt'].shape, dtype=np.float32),
                'feat_distance': np.zeros(self.observation_space['feat_distance'].shape, dtype=np.float32),
            }

        game_part = cv2.resize(info.game_part, self.target_size, interpolation=cv2.INTER_AREA)
        mask_shape = game_part.shape[:2]

        return {
            'image': np.concatenate([
                game_part,
                np.expand_dims(get_mask_by_grid_type(mask_shape, info.grid, game_const.GridType.UNKNOWN), axis=-1),
                np.expand_dims(get_mask_by_grid_type(mask_shape, info.grid, game_const.GridType.EMPTY), axis=-1),
                np.expand_dims(get_mask_by_grid_type(mask_shape, info.grid, game_const.GridType.OWN_BODY), axis=-1),
                np.expand_dims(get_mask_by_grid_type(mask_shape, info.grid, game_const.GridType.OWN_HEAD), axis=-1),
                np.expand_dims(get_mask_by_grid_type(mask_shape, info.grid, game_const.GridType.BLUE_BODY), axis=-1),
                np.expand_dims(get_mask_by_grid_type(mask_shape, info.grid, game_const.GridType.BLUE_HEAD), axis=-1),
                np.expand_dims(get_mask_by_grid_type(mask_shape, info.grid, game_const.GridType.PURPLE_BODY), axis=-1),
                np.expand_dims(get_mask_by_grid_type(mask_shape, info.grid, game_const.GridType.PURPLE_HEAD), axis=-1),
                np.expand_dims(get_mask_by_grid_type(mask_shape, info.grid, game_const.GridType.PINK_BODY), axis=-1),
                np.expand_dims(get_mask_by_grid_type(mask_shape, info.grid, game_const.GridType.PINK_HEAD), axis=-1),

                np.expand_dims(get_mask_by_grid_type(mask_shape, info.grid, game_const.GridType.YELLOW_CRYSTAL), axis=-1),
                np.expand_dims(get_mask_by_grid_type(mask_shape, info.grid, game_const.GridType.GREEN_SPEED), axis=-1),
                np.expand_dims(get_mask_by_grid_type(mask_shape, info.grid, game_const.GridType.BLUE_DIAMOND), axis=-1),

                np.expand_dims(get_mask_by_grid_type(mask_shape, info.grid, game_const.GridType.BOMB), axis=-1),
            ], axis=-1),
            'grid': get_one_hot_grid(info.grid),
            'last_action': np.eye(4)[self.last_info.real_direction].astype(np.uint8) if self.last_info is not None and 0 <= self.last_info.real_direction < 4 else np.zeros(4, dtype=np.uint8),
            'feat_direction_cnt': np.array([
                # info.head[0] * 1.0 / game_const.GRID_ROWS,
                # info.head[1] * 1.0 / game_const.GRID_COLS,
                info.can_go_cnt[0] * 1.0 / game_const.GRID_TOTAL_CNT,
                info.can_go_cnt[1] * 1.0 / game_const.GRID_TOTAL_CNT,
                info.can_go_cnt[2] * 1.0 / game_const.GRID_TOTAL_CNT,
                info.can_go_cnt[3] * 1.0 / game_const.GRID_TOTAL_CNT,
                info.can_go_reward_cnt[0] * 1.0 / info.total_reward_cnt,
                info.can_go_reward_cnt[1] * 1.0 / info.total_reward_cnt,
                info.can_go_reward_cnt[2] * 1.0 / info.total_reward_cnt,
                info.can_go_reward_cnt[3] * 1.0 / info.total_reward_cnt,
                info.direction_danger[0],
                info.direction_danger[1],
                info.direction_danger[2],
                info.direction_danger[3],
                info.direction_reward[0],
                info.direction_reward[1],
                info.direction_reward[2],
                info.direction_reward[3],
            ], dtype=np.float32),
            'feat_distance': np.array([
                info.closest_reward_dy_1 * 1.0 / game_const.GRID_ROWS,
                info.closest_reward_dy_2 * 1.0 / game_const.GRID_ROWS,
                info.closest_reward_dx_1 * 1.0 / game_const.GRID_COLS,
                info.closest_reward_dx_2 * 1.0 / game_const.GRID_COLS,
            ], dtype=np.float32),
        }

    def step(
            self,
            action
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        大概50ms
        Args:
            action:

        Returns:

        """
        # 执行动作
        self._take_action(action)
        self._do_save_replay()

        # 循环等待位置发生变化
        while True:
            # 记录当前时间
            current_time = time.time()
            # 获取新截图
            screenshot = self.screen_capturer.get_screenshot()
            # 获取游戏信息
            info = self.analyzer.analyse(screenshot, current_time, self.last_info)

            if self.last_info.head is None or info.head is None:
                if current_time - self.last_info.current_time >= 0.5:
                    break
            elif self.last_info.head != info.head:
                break

            if info.game_over:
                break

            self.screen_capturer.active_window()
            time.sleep(0.01)

        # 观察空间
        obs = self.get_obs(info)
        # 判断游戏结束
        terminated = info.game_over
        # 当前没有 truncated 的情况
        truncated = False
        # 判断奖励
        reward = self._get_reward(info)
        print(f'上一个: 时间: {self.last_info.current_time - self.last_info.start_time:.4f} 坐标: {self.last_info.head} 动作: {self.last_info.direction} 奖励: {reward:.2f} 预测坐标: {self.last_info.predict_head}')
        print(f'当前: 时间: {info.current_time - info.start_time:.4f} 坐标: {info.head}')
        # 记录游戏信息
        self.last_info_2 = self.last_info
        self.last_info = info

        return obs, reward, terminated, truncated, {'0': info}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        """
        重置游戏

        Args:
            seed:
            options:

        Returns:
            observation (ObsType): 观察空间
            info (dictionary): 其它的调试信息
        """
        gym.Env.reset(self, seed=seed, options=options)
        # 一些必要的重置
        self.screen_capturer.reset()

        # 如果start_time不是0 说明已经开始过游戏 这时候需要等待结算画面出现
        need_summary_screen: bool = self.last_info is not None and self.last_info.start_time != 0
        existed_summary_screen: bool = False
        # 重新开始游戏
        while True:
            current_time: float = time.time()
            screenshot = self.screen_capturer.get_screenshot()

            if game_utils.is_in_game(screenshot):
                # 需要结算画面 而结束画面又没有出现
                # 说明这时候是一局结束的末尾 但还没出现结束画面 这时候不能进入下一轮
                if need_summary_screen and not existed_summary_screen:
                    pass
                else:
                    # 上一帧的游戏信息
                    self.last_info = self.get_init_game_info(current_time, screenshot=screenshot, last_info=None)
                    # 上上一帧的游戏信息
                    self.last_info_2 = self.get_init_game_info(current_time, screenshot=screenshot, last_info=self.last_info)

                    obs = self.get_obs(self.last_info)
                    self.save_idx = 0
                    return obs, {'0': self.last_info}
            else:
                is_summary = game_utils.is_game_over_summary_screen(screenshot)
                existed_summary_screen = existed_summary_screen | is_summary
                if is_summary:
                    self.controller.restart()

            time.sleep(1)

    def _do_save_replay(self) -> None:
        if not self.save_replay:
            return
        self.save_idx += 1

        to_save_dir = os.path.join(self.save_replay_dir, self.last_info.game_round)
        if not os.path.exists(to_save_dir):
            os.mkdir(to_save_dir)

        screenshot_path = os.path.join(to_save_dir, f'{self.save_idx}.png')
        cv2_utils.save_image(self.last_info.screenshot, screenshot_path)

        info_path = os.path.join(to_save_dir, f'{self.save_idx}.json')
        info = {
            'direction': int(self.last_info.direction)
        }
        os_utils.save_json(info, info_path)


class ZzzSnakeReplayEnv(ZzzSnakeEnv):

    def __init__(self):
        pass


def get_mask_by_grid_type_2(shape: tuple[int, int], grid: list[list[int]], grid_type: int) -> np.ndarray:
    """
    根据网格值 绘制对应的原图掩码
    Args:
        shape: 原图 高*宽
        grid: 网格
        grid_type: 目标网格类型

    Returns:
        原图掩码
    """
    row_height: float = shape[0] * 1.0 / game_const.GRID_ROWS
    col_width: float = shape[1] * 1.0 / game_const.GRID_COLS
    mask = np.zeros(shape, dtype=np.uint8)
    for row in range(game_const.GRID_ROWS):
        for col in range(game_const.GRID_COLS):
            if grid[row][col] != grid_type:
                continue
            start_x: int = int(col * col_width)
            end_x: int = int((col + 1) * col_width)
            start_y: int = int(row * row_height)
            end_y: int = int((row + 1) * row_height)
            mask[start_y:end_y, start_x:end_x] = 255
    return mask


def get_mask_by_grid_type(shape: tuple[int, int], grid: NDArray[np.uint8], grid_type: int) -> NDArray[np.uint8]:
    """
    根据网格值 绘制对应的原图掩码
    Args:
        shape: 原图 高*宽
        grid: 网格 (NDArray[np.uint8] 类型)
        grid_type: 目标网格类型

    Returns:
        原图掩码
    """
    row_height: float = shape[0] * 1.0 / game_const.GRID_ROWS
    col_width: float = shape[1] * 1.0 / game_const.GRID_COLS
    mask = np.zeros(shape, dtype=np.uint8)

    # 使用 NumPy 的布尔索引和切片来替代循环
    rows, cols = np.where(grid == grid_type)
    for row, col in zip(rows, cols):
        start_x: int = int(col * col_width)
        end_x: int = int((col + 1) * col_width)
        start_y: int = int(row * row_height)
        end_y: int = int((row + 1) * row_height)
        mask[start_y:end_y, start_x:end_x] = 255

    return mask


def get_one_hot_grid(grid: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """
    根据网格值 绘制对应的one hot编码
    平均1.4ms
    Args:
        grid: 网格
    Returns:
        one hot编码
    """
    oh = np.zeros((game_const.GRID_ROWS, game_const.GRID_COLS, len(game_const.GridType)), dtype=np.uint8)
    for grid_type in game_const.GridType:
        grid_where = np.where(grid == grid_type)
        oh[grid_where[0], grid_where[1], grid_type] = 1

    return oh


def __debug_get_obs():
    env = ZzzSnakeEnv()
    from zzz_snake_gym import debug_utils
    screenshot = debug_utils.get_debug_image('_1745150194431')
    info = env.analyzer.analyse(screenshot, time.time(), None)
    start_time = time.time()
    for _ in range(100):
        obs = env.get_obs(info)
    print(time.time() - start_time)
    print(obs['feat'])
    image = obs['image']
    cv2_utils.show_image(image[:, :, :3], win_name='game_part')
    cv2_utils.show_image(image[:, :, 3], win_name='unknown')
    cv2_utils.show_image(image[:, :, 4], win_name='empty')
    cv2_utils.show_image(image[:, :, 5], win_name='own_body')
    cv2_utils.show_image(image[:, :, 6], win_name='own_head')
    cv2_utils.show_image(image[:, :, 7], win_name='blue_body')
    cv2_utils.show_image(image[:, :, 8], win_name='blue_head')
    cv2_utils.show_image(image[:, :, 9], win_name='purple_body')
    cv2_utils.show_image(image[:, :, 10], win_name='purple_head')
    cv2_utils.show_image(image[:, :, 11], win_name='pink_body')
    cv2_utils.show_image(image[:, :, 12], win_name='pink_head')
    cv2_utils.show_image(image[:, :, 13], win_name='yellow_crystal')
    cv2_utils.show_image(image[:, :, 14], win_name='green_speed')
    cv2_utils.show_image(image[:, :, 15], win_name='blue_diamond')
    cv2_utils.show_image(image[:, :, 16], win_name='bomb')
    cv2.waitKey(1)
    cv2.waitKey(0)


if __name__ == '__main__':
    __debug_get_obs()