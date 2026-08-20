"""
截图管理器
负责：质量过滤、ROI 校验、时间间隔去重、文件命名、存储管理
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta


class CaptureManager:
    """管理截图保存策略，包含去重和存储控制"""

    def __init__(self, config: dict):
        self.output_dir = Path(config["capture"]["output_dir"])
        self.interval = timedelta(seconds=config["capture"]["interval_seconds"])
        self.min_height = config["capture"]["min_person_height_px"]
        self.save_with_bbox = config["capture"]["save_with_bbox"]
        self.max_folder_bytes = config["capture"]["max_daily_folder_gb"] * 1024 ** 3

        self.output_dir.mkdir(parents=True, exist_ok=True)
        # 记录每个网格区域最后截图时间
        self._last_capture: dict[str, datetime] = {}

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
            return True  # ROI 未启用，全画面检测

        h, w = frame_shape[:2]
        points = np.array([
            (int(p[0] * w), int(p[1] * h))
            for p in roi_config["points"]
        ], dtype=np.int32)

        # pointPolygonTest: >=0 表示在多边形内或边上
        return cv2.pointPolygonTest(points, center, False) >= 0

    def _zone_key(self, center: tuple, frame_shape: tuple, grid: int = 3) -> str:
        """
        将画面划分为网格，同网格内视为同一区域
        用于简单去重：同一区域短时间内只截图一次
        """
        h, w = frame_shape[:2]
        gx = min(int(center[0] / w * grid), grid - 1)
        gy = min(int(center[1] / h * grid), grid - 1)
        return f"{gx}-{gy}"

    def _cleanup_if_needed(self, folder: Path):
        """如果单日文件夹总大小超过限制，删除最旧的文件"""
        total = sum(f.stat().st_size for f in folder.iterdir() if f.is_file())
        files = sorted(folder.iterdir(), key=lambda p: p.stat().st_mtime)

        while total > self.max_folder_bytes and files:
            f = files.pop(0)
            if f.is_file():
                total -= f.stat().st_size
                f.unlink()
                print(f"[Capture] 清理旧文件: {f.name}")

    def _next_filename(self, folder: Path, timestamp: str) -> Path:
        """生成不重复的文件名，格式: HH-MM-SS_001.jpg"""
        idx = 1
        while True:
            filename = f"{timestamp}_{idx:03d}.jpg"
            filepath = folder / filename
            if not filepath.exists():
                return filepath
            idx += 1

    # ------------------- 公共方法 -------------------

    def should_capture(self, detection: dict, frame_shape: tuple, roi_config: dict) -> bool:
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
        now = datetime.now()

        if zone in self._last_capture:
            if now - self._last_capture[zone] < self.interval:
                return False  # 间隔太短，跳过

        self._last_capture[zone] = now
        return True

    def save(self, frame: np.ndarray, detection: dict, draw_func=None) -> Path | None:
        """
        保存截图到文件

        Args:
            frame: 原始视频帧
            detection: 检测信息字典
            draw_func: 可选的画框函数

        Returns:
            保存的文件路径，失败返回 None
        """
        today_dir = self._get_today_dir()
        self._cleanup_if_needed(today_dir)

        timestamp = datetime.now().strftime("%H-%M-%S")
        filepath = self._next_filename(today_dir, timestamp)

        # 决定保存原图还是带框图
        if self.save_with_bbox and draw_func:
            img = draw_func(frame, [detection])
        else:
            img = frame

        # 高质量 JPEG 保存
        success = cv2.imwrite(str(filepath), img, [cv2.IMWRITE_JPEG_QUALITY, 95])

        if success:
            print(f"[Capture] ✅ 已保存: {filepath}")
            return filepath
        else:
            print(f"[Capture] ❌ 保存失败: {filepath}")
            return None
