import os

from dotenv import load_dotenv

from zzz_snake_gym import os_utils

# Load environment variables
load_dotenv(
    os.path.join(
        os_utils.get_workspace_dir(),
        '.env'
    )
)

# 训练参数
ENV_DOWN_SCALE = int(os.getenv('ENV_DOWN_SCALE', '8'))

REPLAY_BUFFER_SAVE_EPISODE = int(os.getenv('REPLAY_BUFFER_SAVE_EPISODE', '100'))

OFFLINE_TRAIN_TIMES_PER_BUFFER = int(os.getenv('OFFLINE_TRAIN_TIMES_PER_BUFFER', '100'))  # 离线训练 在每个buffer上的训练次数
OFFLINE_TRAIN_GRADIENT_STEPS = int(os.getenv('BUFFER_TRAIN_GRADIENT_STEPS', '100'))  # 离线训练 每次训练的 采样/更新 次数

ENV_N_STACK = int(os.getenv('ENV_N_STACK', '4'))  # 堆叠帧数
TRAIN_SAVE_GAME_RECORD = os.getenv('TRAIN_SAVE_GAME_RECORD') == '1'  # 普通训练 是否保存游戏记录
TRAIN_BUFFER_SIZE = int(os.getenv('TRAIN_BUFFER_SIZE', '5000'))  # 普通训练 缓存大小
TRAIN_BATCH_SIZE = int(os.getenv('TRAIN_BATCH_SIZE', '200'))  # 普通训练 每次更新前的采样数量
TRAIN_N_STEPS = int(os.getenv('TRAIN_N_STEPS', '10'))  # 普通训练 总共使用多少步的数据进行训练
TRAIN_GRADIENT_STEPS = int(os.getenv('TRAIN_GRADIENT_STEPS', '10'))  # 普通训练 每次训练进行多少次梯度下降
