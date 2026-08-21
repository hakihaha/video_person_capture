

## Codely Structured Memories

### User
- [2026-08-21 09:19:19] [user] 开发硬件为 RTX 4060Ti 16G 显存，使用 yolov8n 模型，显存充裕可支持多视频并行处理。

### Feedback
- [2026-08-21 09:01:36] [feedback] main_local.py 与 main.py 保持分离，不要合并。**Why:** 用户明确说"不用改"，main_local.py 作为单视频调试版独立维护。**How to apply:** 涉及 main_local.py 和 main.py 的去重/合并建议一律跳过。
- [2026-08-21 09:01:36] [feedback] 用户习惯手写变更日志，每次代码改动后需要简洁的改动摘要供其记录。**How to apply:** 完成一批改动后主动提供结构化摘要（文件名 + 改动点 + 原因）。
- [2026-08-21 09:19:19] [feedback] 不要删除任何文件——包括截图输出和视频文件。**Why:** 用户明确说"截图也不要删掉"。**How to apply:** 移除所有自动清理逻辑（_cleanup_if_needed、unlink、max_daily_folder_gb），不自动删除任何产物。
- [2026-08-21 09:19:19] [feedback] 不要静默 OpenCV/FFmpeg 日志。**Why:** 用户明确说"不要静默日志"。**How to apply:** 不设置 OPENCV_LOG_LEVEL=SILENT 等环境变量，HEVC "Could not find ref" 等警告保留即可。
- [2026-08-21 09:42:25] [feedback] 进度条要动态单行刷新，不要每行打印更新。**Why:** 用户要求 tqdm 原地刷新效果。**How to apply:** 处理期间所有输出用 tqdm.write() 而非 print()；video_reader 中不要 print；多线程用单条共享进度条 + pbar_lock 原子更新（不要用 position 多行，会互相覆盖）。模型加载必须在 tqdm 创建之前完成，避免 YOLO 摘要输出打断进度条。不要用 os.dup2 重定向 stdout 来抑制输出——多线程下会导致死锁。


### Project

### Reference

