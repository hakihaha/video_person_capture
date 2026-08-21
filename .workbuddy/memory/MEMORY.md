# video_person_capture 项目记忆

## 优化路线图（2026-08-21 定义）

规范文档：C:\Users\HP\Downloads\基于YOLOv8n监控人员检测系统-技术落地文档.md
（建议落为项目内 SPEC.md 作为活文档，待用户确认）

5 项核心优化（按风险排序）：
1. ROI 裁剪后送模型（左下角）+ 坐标映射回 2K 全图 — 低风险，纯逻辑
2. 动态自适应跳帧（连续 N 次无人→拉大步长；检出人→回退基础间隔）— 低风险，针对夜间 80% 空镜头
3. 结构化输出（时间戳/框坐标/无人段区间）— 低风险
4. 多进程并行（非多线程！每进程独立 YOLO 实例 ~1.5GB，16G 显存 2-3 进程）— 中风险，Windows spawn
5. 硬解码 + 720P 输出 — 高风险，整份文档性能承诺的支点；实用路径=ffmpeg 子进程 `-hwaccel cuda`；软解降级兜底

## 关键技术判定
- 多进程 vs 多线程：文档选多进程是对的（PyTorch 推理虽释放 GIL 但多进程隔离更稳）
- 硬解码：pip opencv-python 不带 NVDEC；cv2.cudacodec 需自编译 OpenCV
- 性能指标前提：2hr≤6min 仅硬解码+夜间动态跳帧叠加成立；250路/3-4hr 依赖夜间占比；文末"AI 生成"注与"实测"标题冲突，指标可能为推算值
- 动态跳帧安全约束：无人态步长上限 ≤ 人员最短出现时长×fps

## 开发约定（CODELY.md，硬性）
- 不删任何文件含截图；不静默 OpenCV/FFmpeg 日志
- 进度条 tqdm 单行刷新 + pbar_lock，不用 position 多行
- main_local.py 与 main.py 保持分离，不合并
- 模型用 yolov8n（非 yolov8x），RTX 4060Ti 16G
- 改动后给结构化摘要（文件名+改动点+原因）

## 当前代码现状
- detector.py: 逐帧推理，已 FP16+预热，缺 model.fuse() 和批量推理
- video_reader.py: cv2.VideoCapture + resize 960，固定 skip_frames=24
- capture_manager.py: 异步写盘（优于参考脚本同步写），含遗留 _cleanup_if_needed 删除逻辑（与"不删文件"约定冲突）
- main.py: 单进程串行，task_status.json 断点续跑（比参考脚本 txt 更完善）
- config.yaml: model=yolov8n, conf=0.3, ROI=左半边, resize=960, skip=24
- 参考脚本 D:\bak\detect_save_people2.py: 批量推理+fuse+多线程并行+静默日志，但同步写盘+txt 续跑更弱
