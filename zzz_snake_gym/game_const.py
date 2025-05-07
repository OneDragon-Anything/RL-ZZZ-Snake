# 蛇对蛇区域
from enum import IntEnum

import numpy as np

GAME_RECT: tuple[int, int, int, int] = (485, 201, 1435, 1031)

# 分数坐标 纵坐标 共用
SCORE_Y: tuple[int, int] = (57, 115)
# 分数坐标 横坐标 每个数字一个坐标
SCORE_X_LIST: list[tuple[int, int]] = [
    (766, 867),
    (810, 851),
    (854, 895),
    (897, 938),
    (941, 982),
    (984, 1025),
    (1028, 1069),
    (1071, 1112),
    (1115, 1156),
    (1158, 1199),
    (1202, 1243),
    (1245, 1286),
    (1289, 1330),
    # (1332, 1373),  # 最后2为一定是0 就不匹配了
    # (1376, 1417),
]

# 分数模板名称
SCORE_TEMPLATE_NAME: list[str] = [
    f'score_{i}'
    for i in range(10)
]


# 画面网格
GRID_ROWS: int = 25
GRID_COLS: int = 29
GRID_TOTAL_CNT: int = GRID_ROWS * GRID_COLS


# 网格类型
class GridType(IntEnum):
    UNKNOWN = 0
    EMPTY = 1
    OWN_HEAD = 2
    OWN_BODY = 3
    BLUE_HEAD = 4
    BLUE_BODY = 5
    PURPLE_HEAD = 6
    PURPLE_BODY = 7
    PINK_HEAD = 8
    PINK_BODY = 9
    GOLD_HEAD = 10
    GOLD_BODY = 11
    YELLOW_CRYSTAL = 12
    GREEN_SPEED = 13
    BLUE_DIAMOND = 14
    GOLD_STAR = 15
    BOMB = 16
    GREY_STONE = 17


# 颜色定义
EMPTY_HSV_RANGE = [ np.array([85, 70, 185], dtype=np.uint8), np.array([97, 250, 245], dtype=np.uint8) ]  # 空白区域

OWN_HEAD_HSV_RANGE = [ np.array([4, 51, 255], dtype=np.uint8), np.array([5, 220, 255], dtype=np.uint8) ]  # 自己的头
OWN_HEAD_EYE_HSV_RANGE = [ np.array([0, 0, 255], dtype=np.uint8), np.array([0, 0, 255], dtype=np.uint8) ]  # 自己的头眼睛
OWN_BODY_HSV_RANGE = [ np.array([4, 51, 255], dtype=np.uint8), np.array([15, 255, 255], dtype=np.uint8) ]  # 自己的身体 包括头
BLUE_HEAD_HSV_RANGE = [ np.array([126, 178, 255], dtype=np.uint8), np.array([127, 184, 255], dtype=np.uint8) ]  # 蓝蛇的头
BLUE_BODY_HSV_RANGE = [ np.array([126, 135, 255], dtype=np.uint8), np.array([130, 184, 255], dtype=np.uint8) ]  # 蓝蛇的身体 包括头
PURPLE_HEAD_HSV_RANGE = [ np.array([150, 228, 255], dtype=np.uint8), np.array([150, 255, 255], dtype=np.uint8) ]  # 紫蛇的头
PURPLE_BODY_HSV_RANGE = [ np.array([133, 135, 255], dtype=np.uint8), np.array([150, 255, 255], dtype=np.uint8) ]  # 紫蛇的身体 包括头
PINK_HEAD_HSV_RANGE = [ np.array([0, 255, 255], dtype=np.uint8), np.array([0, 255, 255], dtype=np.uint8) ]  # 粉蛇的头
PINK_HEAD_HSV_RANGE_2 = [ np.array([180, 255, 255], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8) ]  # 粉蛇的头
PINK_BODY_HSV_RANGE = [ np.array([155, 153, 255], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8) ]  # 粉蛇的身体 包括头
GOLD_HEAD_HSV_RANGE = [ np.array([30, 255, 255], dtype=np.uint8), np.array([30, 255, 255], dtype=np.uint8) ]  # 金蛇的头
GOLD_BODY_HSV_RANGE = [ np.array([17, 255, 255], dtype=np.uint8), np.array([30, 255, 255], dtype=np.uint8) ]  # 金蛇的身体 包括头

YELLOW_CRYSTAL_HSV_RANGE = [ np.array([18, 0, 255], dtype=np.uint8), np.array([25, 192, 255], dtype=np.uint8) ]  # 黄色水晶
GREEN_SPEED_HSV_RANGE = [ np.array([71, 114, 229], dtype=np.uint8), np.array([75, 192, 255], dtype=np.uint8) ]  # 绿色加速
BLUE_DIAMOND_HSV_RANGE = [ np.array([85, 166, 204], dtype=np.uint8), np.array([90, 217, 235], dtype=np.uint8) ]  # 蓝色钻石
GOLD_STAR_HSV_RANGE = [ np.array([15, 216, 229], dtype=np.uint8), np.array([25, 230, 243], dtype=np.uint8) ]  # 无敌金星

BOMB_HSV_RANGE = [ np.array([0, 40, 165], dtype=np.uint8), np.array([0, 166, 230], dtype=np.uint8) ]  # 炸弹区域
BOMB_HSV_RANGE_2 = [ np.array([170, 40, 165], dtype=np.uint8), np.array([180, 166, 230], dtype=np.uint8) ]  # 炸弹区域2
GREY_STONE_HSV_RANGE = [ np.array([114, 25, 136], dtype=np.uint8), np.array([119, 60, 204], dtype=np.uint8) ]  # 灰色石头
