"""
固定视角视频人物检测抓拍系统 - 主程序
【批量递归遍历 + 断点续跑 + 批量推理 + 多进程并行 + ROI裁剪 + 动态跳帧 + 结构化输出】
所有批量路径配置统一放在 config.yaml -> batch_task
"""
import os
import sys
import yaml
import cv2
import signal
import json
import copy
import math
import multiprocessing
from pathlib import Path
from detector import PersonDetector
from capture_manager import CaptureManager
from video_reader import VideoReader
from result_recorder import ResultRecorder
import numpy as np


# ============================================================
#  打包路径工具：所有相对资源统一解析到 exe 所在目录
#  本地运行时解析到源码目录；打包后解析到 exe 旁边
# ============================================================

def resource_path(rel: str) -> str:
    """把相对路径解析到程序真实所在目录（打包/本地通用）"""
    if not isinstance(rel, str) or not rel:
        return rel
    if os.path.isabs(rel):
        return rel
    if getattr(sys, "frozen", False):          # 打包成 exe 后
        base = os.path.dirname(sys.executable)  # exe 所在目录
    else:                                       # 本地 Python 调试
        base = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(base, rel))


# ============================================================
#  ROI 裁剪辅助函数
# ============================================================

def _compute_roi_crop(roi_cfg: dict, frame_shape: tuple) -> tuple | None:
    """
    计算 ROI 多边形的外接矩形，用于裁剪后送模型推理。
    裁剪后只检测局部区域，大幅减少计算量。

    Returns:
        (x1, y1, x2, y2) 像素坐标，或 None（ROI 未启用/区域过小）
    """
    if not roi_cfg.get("enabled", False):
        return None

    h, w = frame_shape[:2]
    points = roi_cfg["points"]
    xs = [int(p[0] * w) for p in points]
    ys = [int(p[1] * h) for p in points]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    # 边界约束
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)

    # 区域过小则不裁剪
    if x2 - x1 < 32 or y2 - y1 < 32:
        return None

    return (x1, y1, x2, y2)


def _map_detections_back(detections: list[dict], offset_x: int, offset_y: int):
    """将裁剪后检测到的坐标映射回原始全画面坐标"""
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        det["bbox"] = (x1 + offset_x, y1 + offset_y,
                       x2 + offset_x, y2 + offset_y)
        cx, cy = det["center"]
        det["center"] = (cx + offset_x, cy + offset_y)
        # height 不变


# ============================================================
#  动态自适应跳帧控制器
# ============================================================

class SkipController:
    """
    动态自适应跳帧：
      - 正常态：按 base_skip 间隔检测
      - 连续 N 批无人 → 逐步拉大跳帧（×multiplier），上限 max_skip
      - 检出人 → 立即回退到 base_skip
    """

    def __init__(self, base_skip: int, no_person_threshold: int = 5,
                 sparse_multiplier: float = 2.0, max_skip: int = 100):
        self.base_skip = base_skip
        self.threshold = max(1, no_person_threshold)
        self.multiplier = max(1.0, sparse_multiplier)
        self.max_skip = max_skip
        self.consecutive_empty = 0
        self.current_skip = base_skip

    def update(self, person_detected: bool) -> int:
        """
        每批推理后调用，返回当前应使用的 skip 值。

        Args:
            person_detected: 本批是否检出人员

        Returns:
            更新后的 skip 值
        """
        if person_detected:
            self.consecutive_empty = 0
            self.current_skip = self.base_skip
        else:
            self.consecutive_empty += 1
            if self.consecutive_empty >= self.threshold:
                levels = self.consecutive_empty // self.threshold
                self.current_skip = min(
                    int(self.base_skip * (self.multiplier ** levels)),
                    self.max_skip
                )
        return self.current_skip

    def reset(self):
        """新视频时重置"""
        self.consecutive_empty = 0
        self.current_skip = self.base_skip


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


def _process_batch(frames, detections_per_frame, frame_metas,
                   capture_mgr, detector, roi_cfg, roi_crop,
                   recorder, video_name, log_prefix,
                   vid_key, task_status_path, lock,
                   is_multiprocess: bool) -> bool:
    """
    处理一批帧的检测结果：截图保存 + 结构化记录 + 日志 + 状态保存。
    单进程和多进程共用此逻辑。

    Args:
        frames: 原始帧列表（用于截图保存和预览）
        detections_per_frame: 每帧的检测结果列表（已映射回原图坐标）
        frame_metas: [(frame_idx, pos_msec), ...]
        capture_mgr: 截图管理器
        detector: 检测器（用于 draw_boxes）
        roi_cfg: ROI 配置
        roi_crop: ROI 裁剪区域 (x1,y1,x2,y2) 或 None
        recorder: 结果记录器或 None
        video_name: 视频名
        log_prefix: 日志前缀（如 "[App]" 或 "[W0]"）
        vid_key: 视频标识
        task_status_path: 状态文件路径
        lock: 多进程锁（单进程传 None）
        is_multiprocess: 是否多进程模式

    Returns:
        本批是否检出人员
    """
    person_found = False

    for frame, detections, (frame_idx, pos_msec) in zip(frames, detections_per_frame, frame_metas):
        # 结构化记录
        if recorder:
            recorder.record(frame_idx, pos_msec, detections)

        captured_count = 0
        for det in detections:
            if capture_mgr.should_capture(det, frame.shape, roi_cfg, pos_msec):
                draw_fn = detector.draw_boxes if capture_mgr.save_with_bbox else None
                capture_mgr.save(frame, det, draw_func=draw_fn,
                                 video_name=video_name, video_pos_msec=pos_msec)
                captured_count += 1

        if detections:
            person_found = True

        if frame_idx % 500 == 0:
            skip_info = ""
            print(f"{log_prefix} 帧 #{frame_idx}, 检测 {len(detections)} 人, 截图 {captured_count}{skip_info}")
            if is_multiprocess and lock:
                _save_status_locked(task_status_path, vid_key,
                                    {"status": "in_progress", "frame": frame_idx}, lock)

    return person_found


def _detect_with_roi(detector, frames, roi_crop):
    """
    ROI 裁剪后批量推理，坐标映射回原图。

    Returns:
        list[list[dict]]: 每帧检测结果（坐标已映射到原图）
    """
    if roi_crop:
        ox, oy, ex, ey = roi_crop
        crop_frames = [f[oy:ey, ox:ex] for f in frames]
        all_detections = detector.detect_batch(crop_frames)
        for dets in all_detections:
            _map_detections_back(dets, ox, oy)
        return all_detections
    else:
        return detector.detect_batch(frames)


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
    base_skip = config["video"]["skip_frames"]
    roi_cfg = config.get("roi", {})

    # 动态跳帧配置
    ds_cfg = config.get("video", {}).get("dynamic_skip", {})
    skip_ctrl = None
    if ds_cfg.get("enabled", False):
        skip_ctrl = SkipController(
            base_skip=base_skip,
            no_person_threshold=ds_cfg.get("no_person_threshold", 5),
            sparse_multiplier=ds_cfg.get("sparse_multiplier", 2.0),
            max_skip=ds_cfg.get("max_skip", 100),
        )

    # 结构化输出配置
    result_cfg = config.get("result", {})
    result_enabled = result_cfg.get("enabled", False)
    result_dir = result_cfg.get("output_dir", "./results")

    log_tag = f"[W{worker_id}]"

    for vid_key, video_path_str, start_frame in video_tasks:
        video_name = Path(video_path_str).stem
        capture_mgr.reset_dedup()
        if skip_ctrl:
            skip_ctrl.reset()
        last_frame = start_frame

        # 初始化结果记录器
        recorder = None
        if result_enabled:
            recorder = ResultRecorder(video_name, video_path_str)

        # 计算当前视频的 skip（动态跳帧可能修改）
        current_skip = base_skip if not skip_ctrl else skip_ctrl.current_skip

        try:
            frame_buffer = []
            frame_metas = []
            with VideoReader(video_path_str, resize_w, current_skip, start_frame) as reader:
                if recorder:
                    recorder.fps = reader.fps

                # ROI 裁剪区域（从第一帧的形状计算，同视频内不变）
                roi_crop = None

                for frame_idx, frame in reader:
                    pos_msec = reader.pos_msec
                    frame_buffer.append(frame)
                    frame_metas.append((frame_idx, pos_msec))
                    last_frame = frame_idx

                    # 第一帧时计算 ROI 裁剪区域
                    if roi_crop is None and roi_cfg.get("enabled", False):
                        roi_crop = _compute_roi_crop(roi_cfg, frame.shape)
                        if roi_crop:
                            print(f"{log_tag} ROI 裁剪: {roi_crop[0]},{roi_crop[1]} → {roi_crop[2]},{roi_crop[3]} "
                                  f"(原图 {frame.shape[1]}x{frame.shape[0]})")

                    if len(frame_buffer) >= batch_size:
                        all_detections = _detect_with_roi(detector, frame_buffer, roi_crop)
                        person_found = _process_batch(
                            frame_buffer, all_detections, frame_metas,
                            capture_mgr, detector, roi_cfg, roi_crop,
                            recorder, video_name, log_tag,
                            vid_key, task_status_path, lock, is_multiprocess=True)

                        # 动态跳帧
                        if skip_ctrl:
                            new_skip = skip_ctrl.update(person_found)
                            if new_skip != reader.skip:
                                reader.set_skip(new_skip)
                                print(f"{log_tag} 跳帧调整: {reader.skip} → {new_skip}")

                        frame_buffer.clear()
                        frame_metas.clear()

                # 处理剩余帧
                if frame_buffer:
                    all_detections = _detect_with_roi(detector, frame_buffer, roi_crop)
                    _process_batch(
                        frame_buffer, all_detections, frame_metas,
                        capture_mgr, detector, roi_cfg, roi_crop,
                        recorder, video_name, log_tag,
                        vid_key, task_status_path, lock, is_multiprocess=True)
                    frame_buffer.clear()
                    frame_metas.clear()

            # 保存结构化结果
            if recorder:
                result_path = Path(result_dir) / f"{video_name}.json"
                recorder.save(str(result_path))

            _save_status_locked(task_status_path, vid_key,
                                {"status": "done", "frame": 0}, lock)
            print(f"{log_tag} 完成: {Path(video_path_str).name}")

        except KeyboardInterrupt:
            print(f"{log_tag} 收到中断信号，保存进度后退出")
            _save_status_locked(task_status_path, vid_key,
                                {"status": "in_progress", "frame": last_frame}, lock)
            break
        except Exception as e:
            print(f"{log_tag} 异常: {Path(video_path_str).name}: {e}")
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
        config_full_path = resource_path(config_path)
        with open(config_full_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # 打包后：把 config 里的相对路径统一解析到 exe 所在目录，
        # 避免打包后工作目录不一致导致找不到模型/状态/结果
        self.config["detection"]["model"] = resource_path(self.config["detection"]["model"])
        self.config["result"]["output_dir"] = resource_path(self.config["result"]["output_dir"])
        bt_cfg = self.config["batch_task"]
        bt_cfg["task_status_file"] = resource_path(bt_cfg["task_status_file"])
        bt_cfg["scan_video_root"] = resource_path(bt_cfg["scan_video_root"])

        # 读取批量任务配置
        self.scan_root = Path(bt_cfg["scan_video_root"])
        self.task_status_path = Path(bt_cfg["task_status_file"])
        self.video_suffix_set = set(bt_cfg["video_suffix"])

        # 并发与批量参数
        self.num_workers = bt_cfg.get("num_workers", 1)
        self.batch_size = self.config.get("detection", {}).get("batch_size", 8)

        # 动态跳帧配置
        ds_cfg = self.config.get("video", {}).get("dynamic_skip", {})
        self.skip_ctrl = None
        if ds_cfg.get("enabled", False):
            self.skip_ctrl = SkipController(
                base_skip=self.config["video"]["skip_frames"],
                no_person_threshold=ds_cfg.get("no_person_threshold", 5),
                sparse_multiplier=ds_cfg.get("sparse_multiplier", 2.0),
                max_skip=ds_cfg.get("max_skip", 100),
            )

        # 结构化输出配置
        result_cfg = self.config.get("result", {})
        self.result_enabled = result_cfg.get("enabled", False)
        self.result_dir = result_cfg.get("output_dir", "./results")

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

    # ==================== 单进程模式 ====================

    def process_one_video(self, video_path: Path, vid_key: str, task_status: dict) -> bool:
        """处理单个视频文件（批量推理 + ROI裁剪 + 动态跳帧 + 结构化输出）"""
        cfg = self.config.copy()
        cfg["video_source"] = str(video_path.absolute())
        resize_w = cfg["video"]["resize_width"]
        base_skip = cfg["video"]["skip_frames"]
        roi_cfg = cfg.get("roi", {})
        video_name = video_path.stem

        # 读取上次中断的帧位置
        entry = task_status.get(vid_key, {})
        start_frame = entry.get("frame", 0) if isinstance(entry, dict) else 0

        # 重置状态
        self.capture_mgr.reset_dedup()
        if self.skip_ctrl:
            self.skip_ctrl.reset()
        last_frame = start_frame

        # 初始化结果记录器
        recorder = None
        if self.result_enabled:
            recorder = ResultRecorder(video_name, str(video_path))

        current_skip = base_skip if not self.skip_ctrl else self.skip_ctrl.current_skip

        try:
            frame_buffer = []
            frame_metas = []
            roi_crop = None  # 从第一帧计算

            with VideoReader(cfg["video_source"], resize_w, current_skip, start_frame) as reader:
                if recorder:
                    recorder.fps = reader.fps

                for frame_idx, frame in reader:
                    if not self.running:
                        task_status[vid_key] = {"status": "in_progress", "frame": last_frame}
                        self._save_task_status(task_status)
                        return False

                    pos_msec = reader.pos_msec
                    frame_buffer.append(frame)
                    frame_metas.append((frame_idx, pos_msec))
                    last_frame = frame_idx

                    # 第一帧时计算 ROI 裁剪区域
                    if roi_crop is None and roi_cfg.get("enabled", False):
                        roi_crop = _compute_roi_crop(roi_cfg, frame.shape)
                        if roi_crop:
                            print(f"[App] ROI 裁剪: {roi_crop[0]},{roi_crop[1]} → {roi_crop[2]},{roi_crop[3]} "
                                  f"(原图 {frame.shape[1]}x{frame.shape[0]})")

                    # 攒满一批再推理
                    if len(frame_buffer) >= self.batch_size:
                        all_detections = _detect_with_roi(self.detector, frame_buffer, roi_crop)
                        person_found = _process_batch(
                            frame_buffer, all_detections, frame_metas,
                            self.capture_mgr, self.detector, roi_cfg, roi_crop,
                            recorder, video_name, "[App]",
                            vid_key, None, None, is_multiprocess=False)

                        # 预览窗口（仅单进程模式）
                        if self.config["video"]["preview"]:
                            self._show_preview(frame_buffer[-1], all_detections[-1], roi_cfg)

                        # 动态跳帧
                        if self.skip_ctrl:
                            new_skip = self.skip_ctrl.update(person_found)
                            if new_skip != reader.skip:
                                reader.set_skip(new_skip)
                                print(f"[App] 跳帧调整: {reader.skip} → {new_skip}")

                        frame_buffer.clear()
                        frame_metas.clear()

                # 处理视频末尾剩余帧
                if frame_buffer:
                    all_detections = _detect_with_roi(self.detector, frame_buffer, roi_crop)
                    _process_batch(
                        frame_buffer, all_detections, frame_metas,
                        self.capture_mgr, self.detector, roi_cfg, roi_crop,
                        recorder, video_name, "[App]",
                        vid_key, None, None, is_multiprocess=False)

                    if self.config["video"]["preview"]:
                        self._show_preview(frame_buffer[-1], all_detections[-1], roi_cfg)

                    frame_buffer.clear()
                    frame_metas.clear()

            if not self.running:
                task_status[vid_key] = {"status": "in_progress", "frame": last_frame}
                self._save_task_status(task_status)
                return False

            # 保存结构化结果
            if recorder:
                result_path = Path(self.result_dir) / f"{video_name}.json"
                recorder.save(str(result_path))

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

    def _show_preview(self, frame, detections, roi_cfg):
        """显示预览窗口（仅单进程模式）"""
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
        features = [f"batch_size={self.batch_size}"]
        if self.skip_ctrl:
            features.append(f"动态跳帧(上限{self.skip_ctrl.max_skip})")
        if self.result_enabled:
            features.append("结构化输出")
        print(f"[App] 运行模式: {mode_label} | {' + '.join(features)}")
        print(f"[App] 需要处理视频数量: {len(to_process)}（待处理 {pending_count}，续跑 {resume_count}）")
        print("=" * 60)

        if len(to_process) == 0:
            print("[App] 没有需要处理的视频，程序退出")
            return

        if self.num_workers <= 1:
            # ===== 单进程模式（兼容旧行为 + 批量推理 + ROI + 动态跳帧）=====
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
    multiprocessing.freeze_support()  # 打包成 exe 后 Windows 多进程必需，必须最先执行
    app = Application()
    app.run()
