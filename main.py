"""
固定视角视频人物检测抓拍系统 - 主程序
【批量递归遍历 + 断点续跑 + 批量推理 + 多进程并行】
所有批量路径配置统一放在 config.yaml -> batch_task
"""
import yaml
import cv2
import signal
import json
import copy
import multiprocessing
from pathlib import Path
from detector import PersonDetector
from capture_manager import CaptureManager
from video_reader import VideoReader
import numpy as np


# ============================================================
#  多进程 worker 函数（top-level，Windows spawn 兼容）
# ============================================================

def _get_video_duration(video_path: str) -> float:
    """快速获取视频时长（秒），仅读头部元信息"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    if fps > 0:
        return frame_count / fps
    return 0.0


def _read_status_safe(path: str) -> dict:
    """安全读取状态文件，容错损坏/不存在"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for key, val in raw.items():
            if isinstance(val, str):
                raw[key] = {"status": val, "frame": 0}
        return raw
    except (json.JSONDecodeError, FileNotFoundError):
        return {}


def _save_status_locked(path: str, vid_key: str, entry: dict, lock):
    """带锁写入状态文件（多进程安全）"""
    with lock:
        status = _read_status_safe(path)
        status[vid_key] = entry
        with open(path, "w", encoding="utf-8") as f:
            json.dump(status, f, ensure_ascii=False, indent=2)


def _flush_batch_worker(worker_id, detector, capture_mgr, frame_buffer,
                        video_name, roi_cfg, vid_key, task_status_path, lock):
    """多进程 worker 内部：处理一批帧的检测和截图"""
    frames = [item[1] for item in frame_buffer]
    all_detections = detector.detect_batch(frames)

    for (frame_idx, frame, pos_msec), detections in zip(frame_buffer, all_detections):
        captured_count = 0
        for det in detections:
            if capture_mgr.should_capture(det, frame.shape, roi_cfg, pos_msec):
                draw_fn = detector.draw_boxes if capture_mgr.save_with_bbox else None
                capture_mgr.save(frame, det, draw_func=draw_fn,
                                 video_name=video_name, video_pos_msec=pos_msec)
                captured_count += 1

        if frame_idx % 500 == 0:
            print(f"[W{worker_id}] 帧 #{frame_idx}, 检测 {len(detections)} 人, 截图 {captured_count}")
            _save_status_locked(task_status_path, vid_key,
                                {"status": "in_progress", "frame": frame_idx}, lock)


def _worker_main(worker_id: int, config: dict, video_tasks: list,
                 task_status_path: str, lock):
    """
    多进程工作函数：每个进程独立加载 YOLO 模型，处理分配到的视频列表

    Args:
        worker_id: 进程编号 0/1/2
        config: 完整配置字典（picklable）
        video_tasks: [(vid_key, video_path_str, start_frame), ...]
        task_status_path: 状态文件绝对路径
        lock: multiprocessing.Lock 用于状态文件原子写入
    """
    print(f"[Worker-{worker_id}] 启动，分配 {len(video_tasks)} 个视频")

    # 每个进程独立初始化模型和截图管理器（禁止共享模型对象）
    detector = PersonDetector(config)
    capture_mgr = CaptureManager(config)

    batch_size = config.get("detection", {}).get("batch_size", 8)
    resize_w = config["video"]["resize_width"]
    skip = config["video"]["skip_frames"]
    roi_cfg = config.get("roi", {})

    for vid_key, video_path_str, start_frame in video_tasks:
        video_name = Path(video_path_str).stem
        capture_mgr.reset_dedup()
        last_frame = start_frame

        try:
            frame_buffer = []
            with VideoReader(video_path_str, resize_w, skip, start_frame) as reader:
                for frame_idx, frame in reader:
                    pos_msec = reader.pos_msec
                    frame_buffer.append((frame_idx, frame, pos_msec))
                    last_frame = frame_idx

                    if len(frame_buffer) >= batch_size:
                        _flush_batch_worker(worker_id, detector, capture_mgr, frame_buffer,
                                            video_name, roi_cfg, vid_key, task_status_path, lock)
                        frame_buffer.clear()

                # 处理剩余帧
                if frame_buffer:
                    _flush_batch_worker(worker_id, detector, capture_mgr, frame_buffer,
                                        video_name, roi_cfg, vid_key, task_status_path, lock)
                    frame_buffer.clear()

            _save_status_locked(task_status_path, vid_key,
                                {"status": "done", "frame": 0}, lock)
            print(f"[Worker-{worker_id}] 完成: {Path(video_path_str).name}")

        except KeyboardInterrupt:
            print(f"[Worker-{worker_id}] 收到中断信号，保存进度后退出")
            _save_status_locked(task_status_path, vid_key,
                                {"status": "in_progress", "frame": last_frame}, lock)
            break
        except Exception as e:
            print(f"[Worker-{worker_id}] 异常: {Path(video_path_str).name}: {e}")
            import traceback
            traceback.print_exc()
            _save_status_locked(task_status_path, vid_key,
                                {"status": "error", "frame": last_frame}, lock)

    capture_mgr.close()
    print(f"[Worker-{worker_id}] 全部完成，退出")


# ============================================================
#  Application 主类
# ============================================================

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

        # 并发与批量参数
        self.num_workers = bt_cfg.get("num_workers", 1)
        self.batch_size = self.config.get("detection", {}).get("batch_size", 8)

        # 单进程模式才在主进程加载模型；多进程模式由 worker 各自加载
        if self.num_workers <= 1:
            self.detector = PersonDetector(self.config)
            self.capture_mgr = CaptureManager(self.config)

        self.running = True
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        print("\n[App] 收到终止信号，正在安全退出...")
        self.running = False

    def _load_task_status(self) -> dict:
        """加载断点状态文件，兼容旧格式（纯字符串 -> {status, frame} 字典）"""
        if not self.task_status_path.exists():
            return {}
        with open(self.task_status_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        for key, val in raw.items():
            if isinstance(val, str):
                raw[key] = {"status": val, "frame": 0}
        return raw

    def _save_task_status(self, status: dict):
        """保存任务状态到json"""
        with open(self.task_status_path, "w", encoding="utf-8") as f:
            json.dump(status, f, ensure_ascii=False, indent=2)

    def _cleanup_task_status(self, status: dict, all_videos: list[Path]) -> dict:
        """清理已不存在视频文件的无效任务记录"""
        existing_keys = {str(p.absolute()) for p in all_videos}
        cleaned = {k: v for k, v in status.items() if k in existing_keys}
        removed = len(status) - len(cleaned)
        if removed > 0:
            print(f"[App] 清理了 {removed} 条无效任务记录（视频文件已不存在）")
        return cleaned

    def _scan_all_video(self) -> list[Path]:
        """递归遍历根目录，找出全部视频文件（包含嵌套子文件夹）"""
        video_list = []
        for p in self.scan_root.rglob("*"):
            if p.is_file() and p.suffix.lower() in self.video_suffix_set:
                video_list.append(p)
        video_list = sorted(video_list)
        return video_list

    # ==================== 单进程批量推理 ====================

    def process_one_video(self, video_path: Path, vid_key: str, task_status: dict) -> bool:
        """处理单个视频文件（批量推理版），返回True成功；False失败/中断"""
        cfg = self.config.copy()
        cfg["video_source"] = str(video_path.absolute())
        resize_w = cfg["video"]["resize_width"]
        skip = cfg["video"]["skip_frames"]
        roi_cfg = cfg.get("roi", {})
        video_name = video_path.stem

        # 读取上次中断的帧位置
        entry = task_status.get(vid_key, {})
        start_frame = entry.get("frame", 0) if isinstance(entry, dict) else 0

        # 重置去重状态
        self.capture_mgr.reset_dedup()
        last_frame = start_frame

        try:
            frame_buffer = []
            with VideoReader(cfg["video_source"], resize_w, skip, start_frame) as reader:
                for frame_idx, frame in reader:
                    if not self.running:
                        task_status[vid_key] = {"status": "in_progress", "frame": last_frame}
                        self._save_task_status(task_status)
                        return False

                    # 关键：在读下一帧之前捕获 pos_msec
                    pos_msec = reader.pos_msec
                    frame_buffer.append((frame_idx, frame, pos_msec))
                    last_frame = frame_idx

                    # 攒满一批再推理
                    if len(frame_buffer) >= self.batch_size:
                        self._flush_batch_single(frame_buffer, video_name, roi_cfg,
                                                 vid_key, task_status)
                        frame_buffer.clear()

                # 处理视频末尾剩余帧
                if frame_buffer:
                    self._flush_batch_single(frame_buffer, video_name, roi_cfg,
                                             vid_key, task_status)
                    frame_buffer.clear()

            if not self.running:
                task_status[vid_key] = {"status": "in_progress", "frame": last_frame}
                self._save_task_status(task_status)
                return False

            print(f"[App] 视频处理完成: {video_path.name}")
            task_status[vid_key] = {"status": "done", "frame": 0}
            return True
        except Exception as e:
            print(f"[App] 处理视频异常 {video_path.name} : {str(e)}")
            import traceback
            traceback.print_exc()
            task_status[vid_key] = {"status": "error", "frame": last_frame}
            self._save_task_status(task_status)
            return False

    def _flush_batch_single(self, frame_buffer, video_name, roi_cfg, vid_key, task_status):
        """单进程模式：处理一批帧（检测 + 截图 + 预览 + 状态保存）"""
        frames = [item[1] for item in frame_buffer]
        all_detections = self.detector.detect_batch(frames)

        for (frame_idx, frame, pos_msec), detections in zip(frame_buffer, all_detections):
            captured_count = 0
            for det in detections:
                if self.capture_mgr.should_capture(det, frame.shape, roi_cfg, pos_msec):
                    draw_fn = self.detector.draw_boxes if self.capture_mgr.save_with_bbox else None
                    self.capture_mgr.save(frame, det, draw_func=draw_fn,
                                         video_name=video_name, video_pos_msec=pos_msec)
                    captured_count += 1

            # 预览窗口（仅单进程模式）
            if self.config["video"]["preview"]:
                preview_frame = frame.copy()
                if roi_cfg.get("enabled", False):
                    h, w = preview_frame.shape[:2]
                    roi_points = np.array([
                        (int(p[0] * w), int(p[1] * h))
                        for p in roi_cfg["points"]
                    ], dtype=np.int32)
                    roi_points = roi_points.reshape((-1, 1, 2))
                    cv2.polylines(preview_frame, [roi_points], isClosed=True, color=(0, 0, 255), thickness=3)
                if self.config["capture"]["save_with_bbox"]:
                    preview_frame = self.detector.draw_boxes(preview_frame, detections)
                cv2.imshow("Video Preview", preview_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("[App] 按下q，终止处理")
                    self.running = False

            if frame_idx % 500 == 0:
                print(f"[{video_name}] 帧 #{frame_idx}, 检测到 {len(detections)} 人, 本次截图 {captured_count} 张")
                task_status[vid_key] = {"status": "in_progress", "frame": frame_idx}
                self._save_task_status(task_status)

    # ==================== 多进程模式 ====================

    def _run_multi(self, to_process: list[Path], task_status: dict):
        """多进程模式：按视频时长贪心分配，spawn 子进程各自加载模型"""
        self._save_task_status(task_status)

        # 获取视频时长用于贪心分配
        print("[App] 正在获取视频时长用于任务分配...")
        video_infos = []
        for vf in to_process:
            vid_key = str(vf.absolute())
            entry = task_status[vid_key]
            start_frame = entry.get("frame", 0)
            duration = _get_video_duration(str(vf))
            video_infos.append((vid_key, str(vf.absolute()), start_frame, duration))

        # 按时长降序，贪心分配给当前负载最小的 worker
        video_infos.sort(key=lambda x: x[3], reverse=True)
        worker_tasks = [[] for _ in range(self.num_workers)]
        worker_loads = [0.0] * self.num_workers

        for vid_key, vpath, sf, dur in video_infos:
            min_idx = worker_loads.index(min(worker_loads))
            worker_tasks[min_idx].append((vid_key, vpath, sf))
            worker_loads[min_idx] += dur

        for i, (tasks, load) in enumerate(zip(worker_tasks, worker_loads)):
            print(f"[App] Worker-{i}: {len(tasks)} 个视频, 总时长 {load:.0f}s")

        # 深拷贝配置，关闭预览（多进程下不显示 GUI）
        worker_config = copy.deepcopy(self.config)
        worker_config["video"]["preview"] = False

        # 启动工作进程
        lock = multiprocessing.Lock()
        status_path_str = str(self.task_status_path.absolute())
        processes = []
        for wid in range(self.num_workers):
            if not worker_tasks[wid]:
                print(f"[App] Worker-{wid} 无任务，跳过")
                continue
            p = multiprocessing.Process(
                target=_worker_main,
                args=(wid, worker_config, worker_tasks[wid], status_path_str, lock)
            )
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

    # ==================== 主入口 ====================

    def run(self):
        print("=" * 60)
        print(f"[App] 开始递归扫描目录: {self.scan_root.resolve()}")
        all_videos = self._scan_all_video()
        print(f"[App] 总共扫描到视频文件：{len(all_videos)} 个")
        if len(all_videos) == 0:
            print("[App] 没有找到任何视频文件，程序退出")
            return

        task_status = self._load_task_status()
        task_status = self._cleanup_task_status(task_status, all_videos)

        # 新增视频加入任务列表
        for vid in all_videos:
            vid_key = str(vid.absolute())
            if vid_key not in task_status:
                task_status[vid_key] = {"status": "pending", "frame": 0}

        self._save_task_status(task_status)

        # 待处理 + 中断续跑的视频都需要执行
        to_process = [p for p in all_videos
                      if task_status[str(p.absolute())]["status"] in ("pending", "in_progress")]
        pending_count = sum(1 for p in to_process
                            if task_status[str(p.absolute())]["status"] == "pending")
        resume_count = len(to_process) - pending_count

        mode_label = f"多进程 x{self.num_workers}" if self.num_workers > 1 else "单进程"
        print(f"[App] 运行模式: {mode_label} | 批量推理: batch_size={self.batch_size}")
        print(f"[App] 需要处理视频数量: {len(to_process)}（待处理 {pending_count}，续跑 {resume_count}）")
        print("=" * 60)

        if len(to_process) == 0:
            print("[App] 没有需要处理的视频，程序退出")
            return

        if self.num_workers <= 1:
            # ===== 单进程模式（兼容旧行为 + 批量推理）=====
            for idx, video_file in enumerate(to_process):
                if not self.running:
                    break
                vid_key = str(video_file.absolute())
                entry = task_status[vid_key]
                start_frame = entry.get("frame", 0)
                status_label = "续跑" if entry["status"] == "in_progress" else "处理"
                print(f"\n========= 任务进度 {idx+1}/{len(to_process)} | {status_label}: {video_file.name} (从第 {start_frame} 帧) =========\n")

                self.process_one_video(video_file, vid_key, task_status)
                self._save_task_status(task_status)

            cv2.destroyAllWindows()
            self.capture_mgr.close()
        else:
            # ===== 多进程模式 =====
            self._run_multi(to_process, task_status)

        if self.running:
            print("\n[App] 全部待处理视频执行完毕！")
        else:
            print("\n[App] 程序被手动终止，下次运行继续处理未完成任务")
        print(f"任务记录保存在: {self.task_status_path.resolve()}")


if __name__ == "__main__":
    app = Application()
    app.run()
