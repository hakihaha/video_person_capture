"""
截图管理器
负责：质量过滤、ROI 校验、时间间隔去重、文件命名、存储管理
支持异步写入和视频时间戳
"""

import cv2
import time
import threading
import queue
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta


class CaptureManager:
    """管理截图保存策略，包含去重和存储控制"""

    def __init__(self, config: dict):
        self.output_dir = Path(config["capture"]["output_dir"])
        self.interval_seconds = config["capture"]["interval_seconds"]
        self.min_height = config["capture"]["min_person_height_px"]
        self.save_with_bbox = config["capture"]["save_with_bbox"]
        self.max_folder_bytes = config["capture"]["max_daily_folder_gb"] * 1024 ** 3

        self.output_dir.mkdir(parents=True, exist_ok=True)
        # 记录每个网格区域最后截图时间（视频时间戳，秒）
        self._last_capture: dict[str, float] = {}
        # 文件夹大小缓存，避免每次截图都全量扫描
        self._folder_sizes: dict[Path, int] = {}
        # 异步写入队列和后台线程
        self._write_queue: queue.Queue = queue.Queue()
        self._writer_thread = threading.Thread(target=self._write_loop, daemon=True)
        self._writer_thread.start()

    # ------------------- 异步写入 -------------------

    def _write_loop(self):
        """后台线程：从队列取出图片写入磁盘"""
        while True:
            item = self._write_queue.get()
            if item is None:
                break
            img, filepath, params = item
            success = cv2.imwrite(str(filepath), img, params)
            if success:
                print(f"[Capture] ✅ 已保存: {filepath}")
            else:
                print(f"[Capture] ❌ 保存失败: {filepath}")
            self._write_queue.task_done()

    def close(self):
        """关闭异步写入线程，等待所有挂起的写入完成"""
        self._write_queue.put(None)
        self._writer_thread.join()

    def reset_dedup(self):
        """重置去重状态，在处理新视频时调用"""
        self._last_capture.clear()

    # ------------------- 私有方法 -------------------

    def _get_today_dir(self) -> Path:
        """获取今日输出目录（按日期分文件夹）"""
        today = datetime.now().strftime("%Y-%m-%d")
        path = self.output_dir / today
        path.mkdir(exist_ok=True)
        return path

    def _check_roi(self, center: tuple, roi_config: dict, frame_shape: tuple) -> bool:
        """判断检测中心点是否在 ROI 多边形内"""
        if not roi_config.get("enabled", False):
            return True

        h, w = frame_shape[:2]
        points = np.array([
            (int(p[0] * w), int(p[1] * h))
            for p in roi_config["points"]
        ], dtype=np.int32)

        return cv2.pointPolygonTest(points, center, False) >= 0

    def _zone_key(self, center: tuple, frame_shape: tuple, grid: int = 3) -> str:
        """将画面划分为网格，同网格内视为同一区域"""
        h, w = frame_shape[:2]
        gx = min(int(center[0] / w * grid), grid - 1)
        gy = min(int(center[1] / h * grid), grid - 1)
        return f"{gx}-{gy}"

    def _get_folder_size(self, folder: Path) -> int:
        """全量扫描文件夹大小"""
        return sum(f.stat().st_size for f in folder.iterdir() if f.is_file())

    def _cleanup_if_needed(self, folder: Path):
        """缓存计数器超阈值时才执行全量扫描清理"""
        cached = self._folder_sizes.get(folder, 0)
        if cached <= self.max_folder_bytes:
            return

        # 重新精确统计
        total = self._get_folder_size(folder)
        self._folder_sizes[folder] = total

        if total <= self.max_folder_bytes:
            return

        files = sorted(folder.iterdir(), key=lambda p: p.stat().st_mtime)
        while total > self.max_folder_bytes and files:
            f = files.pop(0)
            if f.is_file():
                size = f.stat().st_size
                f.unlink()
                total -= size
                print(f"[Capture] 清理旧文件: {f.name}")
        self._folder_sizes[folder] = total

    def _next_filename(self, folder: Path, video_name: str, video_pos_msec: float) -> Path:
        """生成不重复的文件名，使用视频时间戳"""
        idx = 1
        while True:
            if video_pos_msec > 0:
                td = timedelta(milliseconds=video_pos_msec)
                hours, remainder = divmod(int(td.total_seconds()), 3600)
                minutes, seconds = divmod(remainder, 60)
                time_str = f"{hours:02d}-{minutes:02d}-{seconds:02d}"
            else:
                time_str = datetime.now().strftime("%H-%M-%S")

            prefix = f"{video_name}_" if video_name else ""
            filename = f"{prefix}{time_str}_{idx:03d}.jpg"
            filepath = folder / filename
            if not filepath.exists():
                return filepath
            idx += 1

    # ------------------- 公共方法 -------------------

    def should_capture(self, detection: dict, frame_shape: tuple, roi_config: dict,
                       video_pos_msec: float = 0.0) -> bool:
        """
        判断是否应该对本次检测进行截图
        依次检查：尺寸过滤 → ROI 过滤 → 时间间隔去重
        """
        # 1. 尺寸过滤：人物高度不足则跳过
        if detection["height"] < self.min_height:
            return False

        # 2. ROI 过滤：不在设定区域内则跳过
        if not self._check_roi(detection["center"], roi_config, frame_shape):
            return False

        # 3. 时间间隔去重：同一网格区域短时间内只截一次
        zone = self._zone_key(detection["center"], frame_shape)
        current_time = video_pos_msec / 1000.0 if video_pos_msec > 0 else time.time()

        if zone in self._last_capture:
            if current_time - self._last_capture[zone] < self.interval_seconds:
                return False

        self._last_capture[zone] = current_time
        return True

    def save(self, frame: np.ndarray, detection: dict, draw_func=None,
             video_name: str = "", video_pos_msec: float = 0.0) -> Path | None:
        """
        保存截图到文件（异步写入队列）

        Args:
            frame: 原始视频帧
            detection: 检测信息字典
            draw_func: 可选的画框函数
            video_name: 视频文件名（用于截图命名溯源）
            video_pos_msec: 视频时间戳（毫秒）

        Returns:
            保存的文件路径
        """
        today_dir = self._get_today_dir()
        self._cleanup_if_needed(today_dir)

        filepath = self._next_filename(today_dir, video_name, video_pos_msec)

        # 决定保存原图还是带框图（非带框模式必须copy，避免异步写入时帧被覆盖）
        if self.save_with_bbox and draw_func:
            img = draw_func(frame, [detection])
        else:
            img = frame.copy()

        # 放入异步写入队列
        self._write_queue.put((img, filepath, [cv2.IMWRITE_JPEG_QUALITY, 95]))

        # 估算写入大小更新缓存计数器
        self._folder_sizes[today_dir] = self._folder_sizes.get(today_dir, 0) + 200 * 1024

        return filepath
