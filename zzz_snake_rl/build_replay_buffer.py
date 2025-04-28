import os

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback

from zzz_snake_gym import os_utils
from zzz_snake_gym.game_record_env import GameRecordLoader, ZzzSnakeGameRecordEnv
from zzz_snake_rl.env_utils import REPLAY_BUFFER_SAVE_EPISODE
from zzz_snake_rl.train import make_train_env


class RecordedDQN(DQN):
    def __init__(self, record_loader: GameRecordLoader, *args, **kwargs):
        DQN.__init__(self, *args, **kwargs)

        self.record_loader: GameRecordLoader = record_loader
    def predict(self, observation, state=None, episode_start=None, deterministic=False):
        """覆盖predict方法，返回预录制的动作而不是模型预测的动作"""

        # 返回动作和状态（如果有的话）
        return self.record_loader.next_data['direction'], state


class ReplayBufferSaver(BaseCallback):
    """定期保存ReplayBuffer的回调"""

    def __init__(self,
                 save_freq: int,
                 replay_buffer_dir: str,
                 verbose=0):
        BaseCallback.__init__(self, verbose)

        self.episode_cnt: int = 0
        self.save_freq = save_freq  # 多少个episode保存一次
        self.replay_buffer_dir = replay_buffer_dir  # 保存路径

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        if self.model.replay_buffer.size() != self.model.replay_buffer.buffer_size:
            # 满了之后再考虑保存
            return

        self.episode_cnt += 1
        if self.episode_cnt % self.save_freq == 0:
            buffer_idx = self.episode_cnt // self.save_freq
            buffer_path = os.path.join(self.replay_buffer_dir, f'{buffer_idx}.pkl')
            self.model.save_replay_buffer(buffer_path)
            if self.verbose > 0:
                print(f'保存 ReplayBuffer 到 {buffer_path}')


def recorder_env() -> ZzzSnakeGameRecordEnv:
    record_loader = GameRecordLoader(
        game_record_dir=os_utils.get_path_under_workspace_dir(['.debug', 'game_record'])
    )
    return ZzzSnakeGameRecordEnv(record_loader)


def build():
    env: ZzzSnakeGameRecordEnv = make_train_env(recorder_env)

    model = RecordedDQN(env.record_loader)

    save_callback = ReplayBufferSaver(
        save_freq=REPLAY_BUFFER_SAVE_EPISODE,
        replay_buffer_dir=os_utils.get_path_under_workspace_dir(['.debug', 'replay_buffer'])
    )

    model.learn(total_timesteps=env.record_loader.total_step, callback=save_callback)


if __name__ == '__main__':
    build()
