import ctypes
from ctypes import wintypes

# 定义窗口状态常量
SW_RESTORE = 9

def find_window_by_title(window_title) -> wintypes.HWND:
    """根据窗口标题查找窗口句柄"""
    hwnd = ctypes.windll.user32.FindWindowW(None, window_title)
    if hwnd == 0:
        raise ctypes.WinError(ctypes.get_last_error()) if ctypes.get_last_error() != 0 else Exception("Window not found")
    return hwnd


def get_pos_by_hwnd(hwnd: wintypes.HWND) -> tuple[int, int, int ,int]:
    """根据窗口句柄获取窗口位置"""
    ctypes.windll.user32.SetProcessDPIAware()  # 设置 DPI 感知 在屏幕有缩放时有用
    client_rect = wintypes.RECT()
    ctypes.windll.user32.GetClientRect(hwnd, ctypes.byref(client_rect))
    left_top_pos = ctypes.wintypes.POINT(client_rect.left, client_rect.top)
    ctypes.windll.user32.ClientToScreen(hwnd, ctypes.byref(left_top_pos))
    return (left_top_pos.x, left_top_pos.y,
            left_top_pos.x + client_rect.right, left_top_pos.y + client_rect.bottom)


def bring_window_to_foreground(hwnd):
    """
    将指定句柄的窗口带到前台并激活

    参数:
        hwnd (int): 窗口句柄

    返回:
        bool: 操作是否成功
    """
    # 首先恢复窗口（如果最小化）
    ctypes.windll.user32.ShowWindow(hwnd, SW_RESTORE)

    # 然后将窗口设为前台窗口
    return ctypes.windll.user32.SetForegroundWindow(hwnd)


if __name__ == "__main__":
    __hwnd = find_window_by_title("绝区零")
    print(__hwnd)
    print(get_pos_by_hwnd(__hwnd))
    bring_window_to_foreground(__hwnd)
