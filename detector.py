"""
人物检测器 - 基于 YOLOv8x（最强模型）
支持 GPU 自动加速
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

        # 设备选择：auto / cuda:0 / cpu
        device = config["detection"]["device"]
        if device == "auto":
            device = "cuda:0" if torch.cuda.is_available() else "cpu"

        model_path = config["detection"]["model"]
        self.model = YOLO(model_path)
        self.model.to(device)
        self.device = device

        # 模型预热（消除首次推理延迟）
        dummy = torch.zeros(1, 3, 640, 640).to(device)
        self.model.predict(dummy, verbose=False)

        # 打印设备信息
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[Detector] ✅ GPU 已启用: {gpu_name}")
        else:
            print(f"[Detector] ⚠️ GPU 不可用，使用 CPU 推理")
        print(f"[Detector] 模型加载完成: {model_path}, 设备: {device}")

    def detect(self, frame: np.ndarray) -> list[dict]:
        """
        对单帧进行人物检测

        返回:
            list[dict]: 每个元素包含:
                - bbox: (x1, y1, x2, y2) 像素坐标
                - conf: 置信度 float
                - class_id: 类别ID int
                - center: (cx, cy) 中心点
                - height: 人物高度像素
        """
        results = self.model(frame, conf=self.conf, classes=self.classes, verbose=False)
        detections = []

        if results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)

            for box, conf, cls_id in zip(boxes, confs, cls_ids):
                x1, y1, x2, y2 = map(int, box)
                detections.append({
                    "bbox": (x1, y1, x2, y2),
                    "conf": float(conf),
                    "class_id": int(cls_id),
                    "center": ((x1 + x2) // 2, (y1 + y2) // 2),
                    "height": y2 - y1
                })
        return detections

    def draw_boxes(self, frame: np.ndarray, detections: list[dict]) -> np.ndarray:
        """在图像上绘制检测框和标签"""
        img = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"person {d['conf']:.2f}"
            # 标签背景
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw, y1), (0, 255, 0), -1)
            cv2.putText(img, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return img
