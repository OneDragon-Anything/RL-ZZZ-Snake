import os
import time

from stable_baselines3 import PPO, DQN, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from zzz_snake_gym import game_utils
from zzz_snake_gym.controller import ZzzSnakeController, KeyMouseZzzSnakeController
from zzz_snake_gym.screen_capturer import ZzzSnakeScreenCapturer, DefaultZzzSnakeScreenCapturer
from zzz_snake_rl import train_utils

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

# 设置参数
TOTAL_TIMESTEPS = 100000  # 总训练步数
SAVE_FREQ = 10000  # 每隔多少步保存一次模型
EVAL_FREQ = 5000  # 每隔多少步评估一次
N_STACK = 4  # 堆叠的帧数


class GradientPauseCallback(BaseCallback):

    def __init__(
            self,
            win_title: str = '绝区零',
            screen_capturer: ZzzSnakeScreenCapturer = None,
            controller: ZzzSnakeController = None,
    ):
        BaseCallback.__init__(self)

        # 截图器 返回游戏画面的截图 RGB格式
        self.screen_capturer: ZzzSnakeScreenCapturer = DefaultZzzSnakeScreenCapturer(win_title=win_title) if screen_capturer is None else screen_capturer
        self.screen_capturer.reset()

        # 控制器
        self.controller: ZzzSnakeController = KeyMouseZzzSnakeController() if controller is None else controller

        self.ever_pause: bool = False

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self):
        """在 n_steps 数据收集完成前"""
        print('收集数据开始 继续游戏')

        while self.ever_pause:
            screenshot = self.screen_capturer.get_screenshot()
            if game_utils.is_in_game(screenshot) or game_utils.is_game_over(screenshot):
                break
            else:
                self.controller.pause()
                time.sleep(0.1)

        return True

    def _on_rollout_end(self) -> bool:
        """在 n_steps 数据收集完成后、梯度计算前调用"""
        print('收集数据结束 暂停游戏')

        screenshot = self.screen_capturer.get_screenshot()
        if not game_utils.is_game_over(screenshot):
            self.controller.pause()
            self.ever_pause = True

        return True  # 继续训练


def train():
    # 创建向量化环境
    env = make_vec_env(
        train_utils.make_env,
        n_envs=1,  # 并行环境数
        vec_env_cls=SubprocVecEnv  # 多进程并行
    )
    # 帧堆叠
    env = VecFrameStack(env, n_stack=N_STACK)

    # 创建回调函数
    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path=train_utils.get_sb3_checkpoint_dir(),
        name_prefix="zzz_snake_model"
    )
    pause_callback = GradientPauseCallback()

    # eval_callback = EvalCallback(
    #     eval_env,
    #     best_model_save_path=LOG_DIR,
    #     log_path=LOG_DIR,
    #     eval_freq=EVAL_FREQ,
    #     deterministic=True,
    #     render=False,
    #     n_eval_episodes=5
    # )

    # 创建模型
    policy = "MultiInputPolicy"
    model = DQN(
        policy,
        env,
        verbose=1,
        buffer_size=1000,
        tensorboard_log=train_utils.get_tensorboard_log_dir(),
        train_freq=(1, 'episode'),
        gradient_steps=10,
    )

    # 训练模型
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[
            checkpoint_callback,
            # pause_callback
        ],
    )

    # 保存最终模型
    model.save(os.path.join(train_utils.get_sb3_model_save_dir(), "final_model"))

    # 关闭环境
    env.close()


if __name__ == "__main__":
    train()