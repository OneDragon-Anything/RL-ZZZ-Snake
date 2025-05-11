import os
from typing import Callable, Dict

import gymnasium as gym
import torch
import torch.nn as nn
from dotenv import load_dotenv
from gymnasium import spaces
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv

from zzz_snake_gym import os_utils
from zzz_snake_gym.env import ZzzSnakeEnv
from zzz_snake_rl import train_utils
from zzz_snake_rl.env_utils import ENV_N_STACK, TRAIN_SAVE_GAME_RECORD, TRAIN_BUFFER_SIZE, TRAIN_BATCH_SIZE, \
    ENV_DOWN_SCALE, TRAIN_GRADIENT_STEPS

# 设置参数
TOTAL_TIMESTEPS = 100000  # 总训练步数
SAVE_FREQ = 10000  # 每隔多少步保存一次模型


# Load environment variables
load_dotenv(
    os.path.join(
        os_utils.get_workspace_dir(),
        '.env'
    )
)


def normal_env():
    return ZzzSnakeEnv(
        scale=ENV_DOWN_SCALE,
        save_game_record=TRAIN_SAVE_GAME_RECORD,
        game_record_dir=os_utils.get_path_under_workspace_dir(['.debug', 'game_record'])
    )


def make_train_env(env_fn: Callable[[], gym.Env]):
    # 创建向量化环境
    env = make_vec_env(env_fn, vec_env_cls=SubprocVecEnv)
    # 帧堆叠
    return VecFrameStack(env, n_stack=ENV_N_STACK)


def train():
    # 创建环境
    env = make_train_env(normal_env)

    # 创建回调函数
    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path=train_utils.get_sb3_checkpoint_dir(),
        name_prefix="dqn"
    )

    to_save_model_path = os.path.join(train_utils.get_sb3_model_save_dir(), "dqn.zip")
    if os.path.exists(to_save_model_path):
        print('继续上次')
        model = DQN.load(to_save_model_path)
        model.set_env(env)
        model.exploration_initial_eps = 0.1  # 继续训练时 不需要有太多的随机
        model.exploration_schedule = get_linear_fn(
            model.exploration_initial_eps,
            model.exploration_final_eps,
            model.exploration_fraction,
        )
        model.target_update_interval = 5000
        model.learning_rate = 5e-5
        model._setup_lr_schedule()
    else:
        print('重新开始')
        # 创建模型
        model = DQN(
            'MultiInputPolicy',
            env,
            verbose=1,
            buffer_size=TRAIN_BUFFER_SIZE,
            batch_size=TRAIN_BATCH_SIZE,
            tensorboard_log=train_utils.get_tensorboard_log_dir(),
            train_freq=(1, 'episode'),
            gradient_steps=TRAIN_GRADIENT_STEPS,
        )

    # 训练模型
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[
            checkpoint_callback,
        ],
        # progress_bar=True,
    )

    # 保存最终模型
    model.save(to_save_model_path)

    # 关闭环境
    env.close()


if __name__ == "__main__":
    train()
