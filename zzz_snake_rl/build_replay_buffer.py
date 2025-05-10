import os

import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

from zzz_snake_gym import os_utils
from zzz_snake_gym.game_record_env import GameRecordLoader, ZzzSnakeGameRecordEnv
from zzz_snake_rl import train_utils
from zzz_snake_rl.env_utils import REPLAY_BUFFER_SAVE_EPISODE, TRAIN_BUFFER_SIZE, TRAIN_BATCH_SIZE
from zzz_snake_rl.train_dqn import make_train_env

RECORD_LOADER = GameRecordLoader(
    game_record_dir=os_utils.get_path_under_workspace_dir(['.debug', 'game_record'])
)

class RecordedDQN(DQN):
    def __init__(self, record_loader: GameRecordLoader, *args, **kwargs):
        DQN.__init__(self, *args, **kwargs)

        self.record_loader: GameRecordLoader = record_loader
    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        """覆盖predict方法，返回预录制的动作而不是模型预测的动作"""

        # 返回动作和状态（如果有的话）
        return np.array([self.record_loader.next_data['press_direction']], dtype=np.int32), state


class ReplayBufferSaver(BaseCallback):
    """定期保存ReplayBuffer的回调"""

    def __init__(self,
                 record_loader: GameRecordLoader,
                 save_freq: int,
                 replay_buffer_dir: str,
                 verbose=0):
        BaseCallback.__init__(self, verbose)

        self.record_loader: GameRecordLoader = record_loader
        self.episode_cnt: int = 0
        self.save_freq = save_freq  # 多少个episode保存一次
        self.replay_buffer_dir = replay_buffer_dir  # 保存路径

    def _on_step(self) -> bool:
        """
        在step后调用 判断如果是一局游戏的结束的话 就尝试保存buffer
        Returns:

        """
        if not self.record_loader.next_episode_end:
            return True
        if self.model.replay_buffer.size() != self.model.replay_buffer.buffer_size:
            # 满了之后再考虑保存
            return True

        buffer_idx: int = -1
        self.episode_cnt += 1
        if self.episode_cnt % self.save_freq == 0:
            buffer_idx = self.episode_cnt // self.save_freq
        elif self.record_loader.is_last_episode:
            buffer_idx = self.episode_cnt // self.save_freq + 1

        if buffer_idx != -1:
            buffer_path = os.path.join(self.replay_buffer_dir, f'{buffer_idx}.pkl')
            self.model.save_replay_buffer(buffer_path)
            if self.verbose > 0:
                print(f'保存 ReplayBuffer 到 {buffer_path}')
        return True


def recorder_env() -> ZzzSnakeGameRecordEnv:
    return ZzzSnakeGameRecordEnv(RECORD_LOADER)


def build():
    env: ZzzSnakeGameRecordEnv = make_train_env(recorder_env)

    model = RecordedDQN(
        RECORD_LOADER,
        policy='MultiInputPolicy',
        env=env,
        verbose=1,
        buffer_size=TRAIN_BUFFER_SIZE,
        tensorboard_log=train_utils.get_tensorboard_log_dir(),
        # 这里的训练没用 下述参数都是用于避免更新参数的
        train_freq=(1, 'episode'),
        batch_size=TRAIN_BATCH_SIZE,
        gradient_steps=10,
    )

    save_callback = ReplayBufferSaver(
        record_loader=RECORD_LOADER,
        save_freq=REPLAY_BUFFER_SAVE_EPISODE,
        replay_buffer_dir=os_utils.get_path_under_workspace_dir(['.debug', 'replay_buffer'])
    )

    model.learn(
        total_timesteps=RECORD_LOADER.total_step,
        # total_timesteps=1000,
        callback=save_callback
    )
    # 保存最终模型
    model.save(os.path.join(train_utils.get_sb3_model_save_dir(), "buffer_pretrain_model"))


if __name__ == '__main__':
    build()
