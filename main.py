"""
固定视角视频人物检测抓拍系统 - 主程序
功能：读取视频 → YOLOv8x 检测人物 → 智能截图 → 保存到日期文件夹
"""

import yaml
import cv2
import signal
import sys
from pathlib import Path

from detector import PersonDetector
from capture_manager import CaptureManager
from video_reader import VideoReader
import numpy as np


class Application:
    """应用主类，协调各模块工作"""

    def __init__(self, config_path: str = "config.yaml"):
        # 加载配置
        config_full_path = Path(__file__).parent / config_path
        with open(config_full_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # 初始化模块
        self.detector = PersonDetector(self.config)
        self.capture_mgr = CaptureManager(self.config)
        self.running = True

        # 注册信号处理（Ctrl+C 优雅退出）
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """捕获系统信号，安全退出"""
        print("\n[App] ⚠️ 收到终止信号，正在安全退出...")
        self.running = False

    def run(self):
        """主循环"""
        source = self.config["video_source"]
        resize_w = self.config["video"]["resize_width"]
        skip = self.config["video"]["skip_frames"]
        preview = self.config["video"].get("preview", True)
        roi_cfg = self.config.get("roi", {})

        print("=" * 50)
        print("🎬 视频人物检测抓拍系统已启动")
        print(f"   视频源: {source}")
        print(f"   输出目录: {self.config['capture']['output_dir']}")
        print(f"   按 Ctrl+C 或 Q 键退出")
        print("=" * 50)
        print("=" * 50)

        with VideoReader(source, resize_w, skip) as reader:
            for frame_idx, frame in reader:
                if not self.running:
                    break

                # ---- 1. 人物检测 ----
                detections = self.detector.detect(frame)

                # ---- 2. 筛选 + 截图 ----
                captured_count = 0
                for det in detections:
                    if self.capture_mgr.should_capture(det, frame.shape, roi_cfg):
                        draw_fn = self.detector.draw_boxes if self.capture_mgr.save_with_bbox else None
                        self.capture_mgr.save(frame, det, draw_func=draw_fn)
                        captured_count += 1

                # ---- 3. 实时预览（调试用）----
                if preview:
                    vis = self.detector.draw_boxes(frame, detections)

                    # 画出 ROI 区域（如果启用）
                    if roi_cfg.get("enabled"):
                        h, w = frame.shape[:2]
                        pts = [
                            (int(p[0] * w), int(p[1] * h))
                            for p in roi_cfg["points"]
                        ]
                        # 关键修复：转为np.int32，适配OpenCV5.0
                        pts_array = [np.array(pts, dtype=np.int32)]
                        cv2.polylines(vis, pts_array, True, (0, 0, 255), 2)
                        cv2.putText(vis, "ROI", (pts[0][0], pts[0][1] - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

                    # 状态信息叠加
                    status_text = f"Frame: {frame_idx} | Persons: {len(detections)} | Captured: {captured_count}"
                    cv2.putText(vis, status_text, (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

                    cv2.imshow("Person Detection", vis)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("[App] 用户按 Q 键退出")
                        break

                # ---- 4. 定期日志 ----
                if frame_idx % 30 == 0:
                    print(f"[App] 帧 #{frame_idx}, 检测到 {len(detections)} 人, 本次截图 {captured_count} 张")

        # 清理
        cv2.destroyAllWindows()
        print("[App] ✅ 已安全退出，所有资源已释放")


if __name__ == "__main__":
    app = Application()
    app.run()
