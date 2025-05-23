import os
from typing import Any, SupportsFloat

from gymnasium.core import ObsType
from stable_baselines3 import DQN
from tqdm import tqdm

from zzz_snake_gym import os_utils
from zzz_snake_gym.env import ZzzSnakeEnv
from zzz_snake_rl import train_utils
from zzz_snake_rl.env_utils import OFFLINE_TRAIN_TIMES_PER_BUFFER, OFFLINE_TRAIN_GRADIENT_STEPS, \
    TRAIN_BUFFER_SIZE, TRAIN_BATCH_SIZE
from zzz_snake_rl.train import make_train_env


class DummayEnv(ZzzSnakeEnv):

    def __init__(self):
        ZzzSnakeEnv.__init__(self)

    def step(
            self,
            action
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        return self.game_over_obs(), 0, False, False, {}

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        return self.game_over_obs(), {}


def dummy_env() -> DummayEnv:
    return DummayEnv()


def train_with_multiple_buffers():
    """
    使用多个replay buffer循环训练模型
    """
    env = make_train_env(dummy_env)
    model = DQN(
        'MultiInputPolicy',
        env,
        verbose=1,
        buffer_size=TRAIN_BUFFER_SIZE,
        tensorboard_log=train_utils.get_tensorboard_log_dir(),
        batch_size=TRAIN_BATCH_SIZE,
        learning_starts=0,  # 立即开始学习，因为我们已经有数据
        gradient_steps=OFFLINE_TRAIN_GRADIENT_STEPS,
    )
    model.learn(total_timesteps=0)  # 调用一次假的训练 会完成一些初始化

    buffer_dir = os_utils.get_path_under_workspace_dir(['.debug', 'replay_buffer'])

    for buffer_name in os.listdir(buffer_dir):
        if not buffer_name.endswith('.pkl'):
            continue
        buffer_path = os.path.join(buffer_dir, buffer_name)
        model.load_replay_buffer(buffer_path)
        for _ in tqdm(range(OFFLINE_TRAIN_TIMES_PER_BUFFER), desc=f'Buffer 训练 {buffer_name}'):
            model.train(gradient_steps=OFFLINE_TRAIN_GRADIENT_STEPS, batch_size=TRAIN_BATCH_SIZE)

    # 保存最终模型
    model.save(os.path.join(train_utils.get_sb3_model_save_dir(), "buffer_pretrain_model"))


if __name__ == "__main__":
    train_with_multiple_buffers()
