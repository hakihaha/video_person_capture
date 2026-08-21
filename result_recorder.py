"""
结构化结果记录器
每个视频生成一份 JSON 结果文件，包含：
  - 人员出现事件（时间戳、帧号、框坐标、置信度）
  - 无人片段时间区间
  - 统计摘要（总时长 / 有人时长 / 无人时长 / 事件数）
"""


class ResultRecorder:
    """记录检测结果，生成结构化输出文件"""

    def __init__(self, video_name: str, video_path: str = "", fps: float = 0.0):
        self.video_name = video_name
        self.video_path = video_path
        self.fps = fps

        self._events: list[dict] = []
        self._no_person_intervals: list[dict] = []
        self._no_person_start: float | None = 0.0   # 视频开头默认无人
        self._no_person_start_frame: int = 0
        self._last_frame_idx: int = 0
        self._last_pos_msec: float = 0.0

    @staticmethod
    def _msec_to_str(msec: float) -> str:
        """毫秒转 HH:MM:SS.mmm 字符串"""
        if msec < 0:
            msec = 0.0
        total_sec = msec / 1000.0
        hours = int(total_sec // 3600)
        minutes = int((total_sec % 3600) // 60)
        seconds = total_sec % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"

    def record(self, frame_idx: int, pos_msec: float, detections: list[dict]):
        """
        记录一帧的检测结果

        Args:
            frame_idx: 帧序号
            pos_msec: 视频内时间戳（毫秒）
            detections: 该帧的检测列表（已映射到原图坐标）
        """
        self._last_frame_idx = frame_idx
        self._last_pos_msec = pos_msec

        if detections:
            # 有人 → 关闭无人区间（如果开着）
            if self._no_person_start is not None:
                self._no_person_intervals.append({
                    "start": self._msec_to_str(self._no_person_start),
                    "end": self._msec_to_str(pos_msec),
                    "start_frame": self._no_person_start_frame,
                    "end_frame": frame_idx,
                    "duration_sec": round((pos_msec - self._no_person_start) / 1000.0, 2),
                })
                self._no_person_start = None

            # 记录每个检测目标
            for det in detections:
                self._events.append({
                    "frame": frame_idx,
                    "time": self._msec_to_str(pos_msec),
                    "bbox": list(det["bbox"]),
                    "conf": round(det["conf"], 3),
                })
        else:
            # 无人 → 如果当前没有无人区间，开启一个
            if self._no_person_start is None:
                self._no_person_start = pos_msec
                self._no_person_start_frame = frame_idx

    def save(self, output_path: str):
        """
        保存结果到 JSON 文件

        Args:
            output_path: 输出文件路径
        """
        # 关闭最后可能未关闭的无人区间
        if self._no_person_start is not None:
            self._no_person_intervals.append({
                "start": self._msec_to_str(self._no_person_start),
                "end": self._msec_to_str(self._last_pos_msec),
                "start_frame": self._no_person_start_frame,
                "end_frame": self._last_frame_idx,
                "duration_sec": round(
                    (self._last_pos_msec - self._no_person_start) / 1000.0, 2),
            })
            self._no_person_start = None

        # 统计摘要
        total_duration = self._last_pos_msec / 1000.0
        no_person_duration = sum(
            iv["duration_sec"] for iv in self._no_person_intervals)
        person_duration = max(0.0, total_duration - no_person_duration)

        result = {
            "video_name": self.video_name,
            "video_path": self.video_path,
            "fps": self.fps,
            "total_frames_processed": self._last_frame_idx,
            "person_events": self._events,
            "no_person_intervals": self._no_person_intervals,
            "summary": {
                "total_duration_sec": round(total_duration, 2),
                "person_duration_sec": round(person_duration, 2),
                "no_person_duration_sec": round(no_person_duration, 2),
                "event_count": len(self._events),
                "no_person_interval_count": len(self._no_person_intervals),
            },
        }

        import json
        from pathlib import Path
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"[Recorder] 结果已保存: {out} "
              f"(事件 {len(self._events)} 条, 无人段 {len(self._no_person_intervals)} 段)")
