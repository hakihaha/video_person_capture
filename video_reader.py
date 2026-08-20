"""
视频源读取器
支持：本地视频文件、RTSP 网络流、USB 摄像头
具备断线重连能力
"""

import cv2
import time
from pathlib import Path


class VideoReader:
    """封装视频源读取，提供迭代器接口"""

    def __init__(self, source, resize_width: int = 1280, skip_frames: int = 0):
        self.source = source
        self.resize_width = resize_width
        self.skip = skip_frames
        self._cap = None
        self._frame_count = 0
        self._is_file = False

    def __enter__(self):
        """上下文管理器进入"""
        # 判断源类型：数字=摄像头，字符串且文件存在=本地文件，否则=RTSP 流
        if str(self.source).isdigit():
            src = int(self.source)
            source_type = "摄像头"
        else:
            src = str(self.source)
            self._is_file = Path(src).is_file()
            source_type = "文件" if self._is_file else "网络流"

        self._cap = cv2.VideoCapture(src)
        if not self._cap.isOpened():
            raise RuntimeError(f"无法打开视频源: {self.source}")

        # 减少缓冲区延迟（仅部分后端有效）
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # 打印源信息
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[Video] 源类型: {source_type}, 分辨率: {w}x{h}, FPS: {fps:.1f}")
        return self

    def __exit__(self, *args):
        """上下文管理器退出，释放资源"""
        if self._cap:
            self._cap.release()
            self._cap = None

    def __iter__(self):
        return self

    def __next__(self):
        """
        迭代返回下一帧

        Returns:
            tuple: (帧序号, 图像数组)
        """
        while True:
            if self._cap is None:
                raise StopIteration

            ret, frame = self._cap.read()
            self._frame_count += 1

            if not ret:
                # 本地文件：读取完毕
                if self._is_file:
                    raise StopIteration
                # 网络流/摄像头：断线重连
                print("[Video] ⚠️ 流中断，5秒后重连...")
                time.sleep(5)
                self.__exit__()
                try:
                    self.__enter__()
                except RuntimeError as e:
                    print(f"[Video] 重连失败: {e}，10秒后再次尝试...")
                    time.sleep(10)
                continue

            # 跳帧处理
            if self.skip > 0 and self._frame_count % (self.skip + 1) != 0:
                continue

            # 缩放处理（加速推理）
            if self.resize_width > 0:
                h, w = frame.shape[:2]
                if w > self.resize_width:
                    ratio = self.resize_width / w
                    frame = cv2.resize(frame, (self.resize_width, int(h * ratio)))

            return self._frame_count, frame
