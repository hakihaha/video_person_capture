"""
固定视角视频人物检测抓拍系统 - 主程序【批量递归遍历+断点续跑版】
所有批量路径配置统一放在 config.yaml -> batch_task
"""
import yaml
import cv2
import signal
import sys
import json
from pathlib import Path
from detector import PersonDetector
from capture_manager import CaptureManager
from video_reader import VideoReader
import numpy as np


class Application:
    def __init__(self, config_path: str = "config.yaml"):
        config_full_path = Path(__file__).parent / config_path
        with open(config_full_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # 读取批量任务配置
        bt_cfg = self.config["batch_task"]
        self.scan_root = Path(bt_cfg["scan_video_root"])
        self.task_status_path = Path(bt_cfg["task_status_file"])
        self.video_suffix_set = set(bt_cfg["video_suffix"])

        self.running = True
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        print("\n[App] ⚠️ 收到终止信号，正在安全退出...")
        self.running = False

    def _load_task_status(self) -> dict:
        """加载断点状态文件，不存在返回空字典"""
        if not self.task_status_path.exists():
            return {}
        with open(self.task_status_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_task_status(self, status: dict):
        """保存任务状态到json"""
        with open(self.task_status_path, "w", encoding="utf-8") as f:
            json.dump(status, f, ensure_ascii=False, indent=2)

    def _scan_all_video(self) -> list[Path]:
        """递归遍历根目录，找出全部视频文件（包含嵌套子文件夹）"""
        video_list = []
        for p in self.scan_root.rglob("*"):
            if p.is_file() and p.suffix.lower() in self.video_suffix_set:
                video_list.append(p)
        video_list = sorted(video_list)
        return video_list

    def process_one_video(self, video_path: Path):
        """处理单个视频文件，返回True成功；False失败/中断"""
        cfg = self.config.copy()
        cfg["video_source"] = str(video_path.absolute())
        resize_w = cfg["video"]["resize_width"]
        skip = cfg["video"]["skip_frames"]
        roi_cfg = cfg.get("roi", {})

        detector = PersonDetector(cfg)
        capture_mgr = CaptureManager(cfg)

        try:
            with VideoReader(cfg["video_source"], resize_w, skip) as reader:
                for frame_idx, frame in reader:
                    if not self.running:
                        print("[App] 用户中断任务")
                        return False

                    detections = detector.detect(frame)
                    captured_count = 0
                    for det in detections:
                        if capture_mgr.should_capture(det, frame.shape, roi_cfg):
                            draw_fn = detector.draw_boxes if capture_mgr.save_with_bbox else None
                            capture_mgr.save(frame, det, draw_func=draw_fn)
                            captured_count += 1

                    if frame_idx % 30 == 0:
                        print(f"[{video_path.name}] 帧 #{frame_idx}, 检测到 {len(detections)} 人, 本次截图 {captured_count} 张")
            print(f"✅ 视频处理完成: {video_path.name}")
            return True
        except Exception as e:
            print(f"❌ 处理视频异常 {video_path.name} : {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        finally:
            del detector
            del capture_mgr

    def run(self):
        print("=" * 60)
        print(f"[App] 开始递归扫描目录: {self.scan_root.resolve()}")
        all_videos = self._scan_all_video()
        print(f"[App] 总共扫描到视频文件：{len(all_videos)} 个")
        if len(all_videos) == 0:
            print("[App] 没有找到任何视频文件，程序退出")
            return

        task_status = self._load_task_status()

        # 新增视频加入任务列表
        for vid in all_videos:
            vid_key = str(vid.absolute())
            if vid_key not in task_status:
                task_status[vid_key] = "pending"

        self._save_task_status(task_status)

        pending_list = [p for p in all_videos if task_status[str(p.absolute())] == "pending"]
        print(f"[App] 需要处理视频数量: {len(pending_list)}，已完成直接跳过")
        print("=" * 60)

        for idx, video_file in enumerate(pending_list):
            if not self.running:
                break
            vid_key = str(video_file.absolute())
            print(f"\n========= 任务进度 {idx+1}/{len(pending_list)} | {video_file.name} =========\n")

            ok = self.process_one_video(video_file)
            if ok:
                task_status[vid_key] = "done"
            else:
                if self.running:
                    task_status[vid_key] = "error"
            self._save_task_status(task_status)

        cv2.destroyAllWindows()
        if self.running:
            print("\n🎉 全部待处理视频执行完毕！")
        else:
            print("\n⚠️ 程序被手动终止，下次运行继续处理未完成任务")
        print(f"任务记录保存在: {self.task_status_path.resolve()}")


if __name__ == "__main__":
    app = Application()
    app.run()