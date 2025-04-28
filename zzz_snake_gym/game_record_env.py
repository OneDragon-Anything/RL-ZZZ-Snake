import os
import time
from typing import SupportsFloat, Any, Optional

import gymnasium as gym
from cv2.typing import MatLike
from gymnasium.core import ObsType

from zzz_snake_gym import cv2_utils, os_utils
from zzz_snake_gym.env import ZzzSnakeEnv


class GameRecordLoader:

    def __init__(
            self,
            game_record_dir: str,
    ):
        """

        Args:
            game_record_dir: 预录制文件的目录
        """
        self.game_record_dir = game_record_dir
        if not os.path.exists(self.game_record_dir):
            raise FileNotFoundError(f"预录制文件目录不存在: {self.game_record_dir}")

        self.total_step: int = 0  # 所有episode的总step数
        self.episode_list: list[str] = []  # 预录制目录下 所有的episode目录
        self.episode_idx: int = 0  # 当前episode下标
        self.step_idx: int = 1  # 当前episode的step下标

        self.next_screenshot: Optional[MatLike] = None
        self.next_data: Optional[dict] = None

    def load_data(self) -> None:
        """
        加载预录制数据
        Returns:
            None
        """
        for episode_name in os.listdir(self.game_record_dir):
            episode_path = os.path.join(self.game_record_dir, episode_name)
            if os.ispath.isdir(episode_path):
                self.episode_list.append(episode_path)
                self.total_step += len(os.listdir(episode_path))

        self.total_step = self.total_step // 2  # png和json各一个
        print(f'总共: {len(self.episode_list)} 个episode, {self.total_step} 个step')

    def next_episode(self) -> None:
        """
        进入下一个episode
        Returns:
            None
        """
        self.episode_idx += 1
        self.step_idx = 1

    def next_step(self) -> None:
        """
        加载预录制的一帧
        Returns:
            None
        """
        screenshot_path = os.path.join(self.episode_list[self.episode_idx], f'{self.step_idx}.png')
        json_path = os.path.join(self.episode_list[self.episode_idx], f'{self.step_idx}.json')
        self.step_idx += 1
        if not os.path.exists(screenshot_path) or not os.path.exists(json_path):
            self.next_screenshot = None
            self.next_data = None
        else:
            self.next_screenshot = cv2_utils.read_image(screenshot_path)
            self.next_data = os_utils.read_json(json_path)


class ZzzSnakeGameRecordEnv(ZzzSnakeEnv):

    r"""
    通过读取预录制的游戏截图和动作 模拟一个gym环境输出
    """

    def __init__(
            self,
            record_loader: GameRecordLoader
    ):
        """

        Args:
            record_loader: 预录制加载器
        """
        ZzzSnakeEnv.__init__(self, save_game_record=False)

        self.record_loader: GameRecordLoader = record_loader

    def step(
            self,
            action
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        # 先执行动作
        self.update_by_action(action)
        ZzzSnakeEnv.update_by_action(self, action)

        # 加载下一帧画面
        self.record_loader.next_step()
        screenshot = self.record_loader.next_screenshot
        screenshot_time = self.record_loader.next_data.get('screenshot_time')
        if screenshot_time is None:
            screenshot_time = time.time()

        # 继续原来的逻辑
        game_info = self.analyzer.analyse(screenshot, screenshot_time, self.last_info)
        return ZzzSnakeEnv.step_with_info(self, game_info)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        gym.Env.reset(self, seed=seed, options=options)
        self.record_loader.next_episode()
        self.record_loader.next_step()

        screenshot = self.record_loader.next_screenshot
        screenshot_time = self.record_loader.next_data.get('screenshot_time')
        if screenshot_time is None:
            screenshot_time = time.time()

        return ZzzSnakeEnv.reset_result(self, screenshot, screenshot_time)
