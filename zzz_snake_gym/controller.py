import time
from enum import IntEnum
from typing import Union

from pynput import keyboard


class ZzzSnakeDirection(IntEnum):

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


class ZzzSnakeController:

    def __init__(self):
        """
        蛇对蛇的控制器
        由子类实现具体操作
        """
        pass

    def move(self, direction: int, press_time: float = 0.01) -> None:
        """
        移动
        """
        pass

    def restart(self) -> None:
        """
        按键J 重启游戏
        Returns:
            无
        """
        pass

    def pause(self) -> None:
        """
        按键ESC 暂停游戏
        Returns:
            无
        """
        pass


class KeyMouseZzzSnakeController(ZzzSnakeController):

    def __init__(self):
        ZzzSnakeController.__init__(self)
        self.keyboard = keyboard.Controller()
        self.move_key: list[keyboard.Key] = [
            keyboard.Key['up'],
            keyboard.Key['down'],
            keyboard.Key['left'],
            keyboard.Key['right'],
        ]  # 移动对应的按键
        self.restart_key: keyboard.KeyCode = keyboard.KeyCode.from_char('j')
        self.pause_key: keyboard.Key = keyboard.Key['esc']

    def press_key(self, key: Union[keyboard.KeyCode, keyboard.Key], press_time: float = 0.02) -> None:
        """
        进行一个按键

        Args:
            key: 按键
            press_time: 持续时间 默认0.01秒

        Returns:
            无
        """
        self.keyboard.press(key)
        if press_time is not None:
            time.sleep(press_time)
            self.keyboard.release(key)

    def move(self, direction: int, press_time: float = 0.01) -> None:
        """
        向某个方向移动

        Args:
            direction: 方向 见 ZzzSnakeDirection
            press_time: 按键持续时间

        Returns:
            无
        """
        self.press_key(self.move_key[direction])

    def restart(self) -> None:
        """
        按键J 重启游戏
        Returns:
            无
        """
        self.press_key(self.restart_key)

    def pause(self) -> None:
        """
        按键ESC 暂停游戏
        Returns:
            无
        """
        self.press_key(self.pause_key, press_time=0.05)
