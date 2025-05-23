import os
from typing import Callable

import gymnasium as gym
from dotenv import load_dotenv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import DummyVecEnv, make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv

from zzz_snake_gym import os_utils
from zzz_snake_gym.env import ZzzSnakeEnv
from zzz_snake_rl import train_utils
from zzz_snake_rl.env_utils import ENV_N_STACK, TRAIN_SAVE_GAME_RECORD, TRAIN_BATCH_SIZE, TRAIN_N_STEPS, ENV_DOWN_SCALE

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
    env = make_vec_env(env_fn)
    # 帧堆叠
    return VecFrameStack(env, n_stack=ENV_N_STACK)


def train():
    # 创建环境
    env = make_train_env(normal_env)

    # 创建回调函数
    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path=train_utils.get_sb3_checkpoint_dir(),
        name_prefix="ppo"
    )

    to_save_model_path = os.path.join(train_utils.get_sb3_model_save_dir(), "ppo.zip")
    if os.path.exists(to_save_model_path):
        model = PPO.load(to_save_model_path)
        model.set_env(env)
    else:
        # 创建模型
        model = PPO(
            'MultiInputPolicy',
            env,
            verbose=1,
            n_steps=TRAIN_N_STEPS,
            batch_size=TRAIN_BATCH_SIZE,
            tensorboard_log=train_utils.get_tensorboard_log_dir(),
        )

    # 训练模型
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[
            checkpoint_callback,
        ],
    )

    # 保存最终模型
    model.save(to_save_model_path)

    # 关闭环境
    env.close()


if __name__ == "__main__":
    train()
