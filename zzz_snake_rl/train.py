import os
import time

from stable_baselines3 import PPO, DQN, SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from zzz_snake_gym import game_utils, os_utils
from zzz_snake_gym.controller import ZzzSnakeController, KeyMouseZzzSnakeController
from zzz_snake_gym.env import ZzzSnakeEnv
from zzz_snake_gym.screen_capturer import ZzzSnakeScreenCapturer, DefaultZzzSnakeScreenCapturer
from zzz_snake_rl import train_utils

from dotenv import load_dotenv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

# 设置参数
TOTAL_TIMESTEPS = 100000  # 总训练步数
SAVE_FREQ = 10000  # 每隔多少步保存一次模型
EVAL_FREQ = 5000  # 每隔多少步评估一次
N_STACK = 4  # 堆叠的帧数


# Load environment variables
load_dotenv()


def make_env():
    return ZzzSnakeEnv(
        save_replay=os.getenv('TRAIN_SAVE_REPLAY') == '1',
        save_replay_dir=os_utils.get_path_under_workspace_dir(['.debug', 'env_replay'])
    )


def train():
    # 创建向量化环境
    env = make_vec_env(
        make_env,
        n_envs=1,  # 并行环境数
        vec_env_cls=SubprocVecEnv,  # 多进程并行
    )
    # 帧堆叠
    env = VecFrameStack(env, n_stack=N_STACK)

    # 创建回调函数
    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path=train_utils.get_sb3_checkpoint_dir(),
        name_prefix="zzz_snake_model"
    )

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
        ],
    )

    # 保存最终模型
    model.save(os.path.join(train_utils.get_sb3_model_save_dir(), "final_model"))

    # 关闭环境
    env.close()


if __name__ == "__main__":
    train()
