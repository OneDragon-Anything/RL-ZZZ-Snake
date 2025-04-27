import time
from ctypes import wintypes
from typing import Optional

import cv2
import mss
import numpy as np
from mss.base import MSSBase

from zzz_snake_gym import ctypes_utils


class ZzzSnakeScreenCapturer:

    def __init__(self):
        """
        负责提供游戏画面的截图
        """
        pass

    def active_window(self) -> None:
        """
        激活游戏窗口
        """
        pass

    def reset(self) -> None:
        """
        环境重置时 需要做的初始化
        """
        pass

    def get_screenshot(self) -> np.array:
        """
        获取游戏截图 必须返回 1920*1080 RGB图片
        由子类实现
        """
        pass


class DefaultZzzSnakeScreenCapturer(ZzzSnakeScreenCapturer):

    def __init__(self, win_title: str = '绝区零'):
        ZzzSnakeScreenCapturer.__init__(self)

        self.win_title: str = win_title
        self.hwnd: Optional[wintypes.HWND] = None  # 窗口句柄
        self.win_pos: Optional[tuple[int, int, int ,int]] = None  # 窗口坐标
        self.sct: Optional[MSSBase] = None  # 截图类
        self.sct_monitor: Optional[dict[str, int]] = None  # 截图用的坐标

    def active_window(self):
        if self.hwnd is not None:
            ctypes_utils.bring_window_to_foreground(self.hwnd)  # 窗口显示在最前方

    def reset(self) -> None:
        """
        环境重置时 需要做的初始化
        """
        self.hwnd = ctypes_utils.find_window_by_title(self.win_title)
        ctypes_utils.bring_window_to_foreground(self.hwnd)  # 窗口显示在最前方
        time.sleep(1)  # 等待一会窗口显示
        self.win_pos = ctypes_utils.get_pos_by_hwnd(self.hwnd)
        if self.sct is not None:
            self.sct.close()
        self.sct = mss.mss()
        self.sct_monitor = {
            "left": self.win_pos[0],
            "top": self.win_pos[1],
            "width": self.win_pos[2] - self.win_pos[0],
            "height": self.win_pos[3] - self.win_pos[1],
        }


    def get_screenshot(self) -> np.array:
        return cv2.cvtColor(np.array(self.sct.grab(self.sct_monitor)), cv2.COLOR_BGRA2RGB)