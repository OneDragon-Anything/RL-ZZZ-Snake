import os

from stable_baselines3 import DQN

from zzz_snake_gym import os_utils
from zzz_snake_rl import train_utils
from zzz_snake_rl.env_utils import OFFLINE_TRAIN_TIMES_PER_BUFFER, OFFLINE_TRAIN_GRADIENT_STEPS, \
    TRAIN_BUFFER_SIZE, TRAIN_BATCH_SIZE
from zzz_snake_rl.train import make_train_env, normal_env


def train_with_multiple_buffers():
    """
    使用多个replay buffer循环训练模型
    """
    env = make_train_env(normal_env)
    model = DQN(
        'MultiInputPolicy',
        env,
        verbose=1,
        buffer_size=TRAIN_BUFFER_SIZE,
        tensorboard_log=train_utils.get_tensorboard_log_dir(),
        batch_size=TRAIN_BATCH_SIZE
    )

    buffer_dir = os_utils.get_path_under_workspace_dir(['.debug', 'replay_buffer'])

    for buffer_name in os.listdir(buffer_dir):
        if not buffer_name.endswith('.pkl'):
            continue
        print(f'加载 buffer {buffer_name}')
        buffer_path = os.path.join(buffer_dir, buffer_name)
        model.load_replay_buffer(buffer_path)
        for _ in range(OFFLINE_TRAIN_TIMES_PER_BUFFER):
            model.train(gradient_steps=OFFLINE_TRAIN_GRADIENT_STEPS, batch_size=model.batch_size)

    # 保存最终模型
    model.save(os.path.join(train_utils.get_sb3_model_save_dir(), "buffer_pretrain_model"))


if __name__ == "__main__":
    train_with_multiple_buffers()
