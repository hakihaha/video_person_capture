"""
人物检测器 - 基于 YOLOv8n
支持 GPU 自动加速、FP16 半精度、层融合(fuse)、批量推理
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO


class PersonDetector:
    """封装 YOLO 检测器，专注于人物检测"""

    def __init__(self, config: dict):
        self.conf = config["detection"]["conf_threshold"]
        self.classes = config["detection"]["classes"]
        self.batch_size = config.get("detection", {}).get("batch_size", 8)

        # 设备选择：auto / cuda:0 / cpu
        device = config["detection"]["device"]
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        model_path = config["detection"]["model"]
        self.model = YOLO(model_path)
        self.model.to(device)

        # 层融合（Conv+BN），减少内核启动开销
        try:
            self.model.fuse()
        except Exception as e:
            print(f"[Detector] fuse() 跳过: {e}")

        # GPU 启用 FP16 半精度
        self.use_half = device.startswith("cuda")
        if self.use_half:
            self.model.half()

        self.device = device

        # 模型预热（消除首次推理延迟）
        dummy = torch.zeros(1, 3, 640, 640).to(device)
        if self.use_half:
            dummy = dummy.half()
        self.model.predict(dummy, verbose=False)

        # 打印设备信息
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[Detector] GPU 已启用: {gpu_name}")
        else:
            print(f"[Detector] GPU 不可用，使用 CPU 推理")
        print(f"[Detector] 模型加载完成: {model_path}, 设备: {device}, batch_size: {self.batch_size}")

    def detect_batch(self, frames: list[np.ndarray]) -> list[list[dict]]:
        """
        批量检测多帧（GPU 批量推理，显著提升吞吐）

        Args:
            frames: 帧图像列表（同尺寸 numpy 数组）

        Returns:
            list[list[dict]]: 每帧的检测结果列表，结构同 detect() 返回值
        """
        if not frames:
            return []

        # half=True 已废弃（Ultralytics 新版），改用初始化时 model.half() 转权重
        # 模型已在 __init__ 中 .half()，Ultralytics 预处理会自动按模型 dtype 转输入
        results = self.model(
            frames,
            conf=self.conf,
            classes=self.classes,
            device=self.device,
            verbose=False,
        )

        all_detections = []
        for result in results:
            detections = []
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                cls_ids = result.boxes.cls.cpu().numpy().astype(int)

                for box, conf, cls_id in zip(boxes, confs, cls_ids):
                    x1, y1, x2, y2 = map(int, box)
                    detections.append({
                        "bbox": (x1, y1, x2, y2),
                        "conf": float(conf),
                        "class_id": int(cls_id),
                        "center": ((x1 + x2) // 2, (y1 + y2) // 2),
                        "height": y2 - y1
                    })
            all_detections.append(detections)
        return all_detections

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        对单帧进行人物检测（兼容旧接口，内部走批量路径）

        Returns:
            list[dict]: 每个元素包含:
                - bbox: (x1, y1, x2, y2) 像素坐标
                - conf: 置信度 float
                - class_id: 类别ID int
                - center: (cx, cy) 中心点
                - height: 人物高度像素
        """
        return self.detect_batch([frame])[0]

    def draw_boxes(self, frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        """在图像上绘制检测框和标签"""
        img = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"person {d['conf']:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw, y1), (0, 255, 0), -1)
            cv2.putText(img, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return img
