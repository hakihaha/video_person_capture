直接复制全部内容，新建文件 README.md 放到项目根目录，一起提交到本地 Git 即可。
markdown
# video_person_capture
YOLOv8‑X 视频人员目标检测项目

## 📋 项目简介
基于 Ultralytics‑YOLOv8 实现本地视频文件的人员识别检测。
优先使用 NVIDIA‑CUDA GPU 硬件加速推理；CPU推理作为降级备用方案，速度较慢。

> 开发硬件：RTX 4060‑Ti 16G

## 📁 目录结构说明
video_person_capture/
├── .venv/ # Python 虚拟环境（Git 忽略，不上传）
├── .idea/ # PyCharm 配置文件（Git 忽略）
├── pycache/ # Python 编译缓存（Git 忽略）
├── output/ # 检测结果输出目录（截图、输出视频，Git 忽略）
├── capture_manager.py # 结果捕获管理模块
├── detector.py # YOLO 检测核心模块
├── main.py # 程序入口主文件
├── video_reader.py # 视频文件读取模块
├── config.yaml # 项目参数配置文件
├── requirements.txt # Python 依赖清单
├── yolov8x.pt # YOLO 官方预训练权重（Git 忽略，运行自动下载）
├── .gitignore # Git 忽略配置
└── README.md # 项目说明文档
plaintext

## ⚙️ 环境部署指南（Windows）
> ⚠️ **安装顺序非常关键！顺序错误会安装到CPU版PyTorch，GPU无法启用**

### 1. 前置条件
- Python 3.13.0
- NVIDIA显卡驱动版本 ≥550.x，支持 CUDA‑12.4
- 查看显卡与CUDA支持版本命令：
```powershell
nvidia-smi
2. 创建并激活虚拟环境
powershell
# 创建虚拟环境
python -m venv .venv

# 激活虚拟环境
.\.venv\Scripts\Activate.ps1
激活成功后终端前缀出现 (.venv)
3. 安装 CUDA‑GPU 版 PyTorch（优先安装！！）
powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
4. GPU 环境有效性验证
powershell
python
python
运行
import torch
print("PyTorch版本:", torch.__version__)
print("CUDA是否可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("显卡名称:", torch.cuda.get_device_name(0))
exit()
✅ 成功标识：版本后缀 +cu124、CUDA 可用 = True
❌ 失败标识：版本后缀 +cpu → GPU 失效，需要卸载清理缓存后重装
5. 安装项目其余依赖
powershell
pip install ultralytics opencv-python
🚀 运行方式
修改 config.yaml，配置视频路径、检测参数
执行主程序入口：
powershell
python main.py
设备选择说明
YOLOv8 默认自动优先调用 CUDA 显卡；如需手动指定设备：
python
运行
# 使用GPU(0号显卡)
model.predict(source="xxx.mp4", device=0)

# 强制降级CPU调试（速度慢）
model.predict(source="xxx.mp4", device="cpu")
⚠️ 显存风险提示
yolov8x 为大型权重，16G 显存刚好适配；高分辨率视频可能触发显存溢出 (OOM)
显存不足解决方案：更换轻量化模型 yolov8m.pt/yolov8s.pt
🛠️ 历史问题记录（踩坑备忘）
故障现象：显卡硬件识别正常，但 torch.cuda.is_available () 返回 False
根源：安装了 CPU 版本 PyTorch，不带 CUDA 运算库
诱因：使用 -f 参数镜像安装时，pip 优先加载本地缓存 CPU 包；先装 ultralytics 会自动拉取 CPU 版 torch
修复步骤：
pip uninstall -y torch torchvision torchaudio
pip cache purge 清理缓存
使用 --index-url 参数强制下载 CUDA 版本 PyTorch
GPU 验证通过后，再安装 YOLO 依赖
📦 环境迁移复现注意事项
requirements.txt 仅作为依赖清单参考！
新环境部署不能直接先执行 pip install -r requirements.txt
正确复现顺序：先手动安装 CUDA 版 PyTorch → 再安装 requirements 内其余依赖
🧩 Git 版本控制说明（本地仓库）
.venv、.idea、pycache、output、*.pt 已配置忽略，不纳入版本管理
当前仅使用本地 Git 仓库，未配置云端远程地址，不会自动上传代码
plaintext

保存之后，你就可以执行
```powershell
git add .
git commit -m "初始版本：YOLOv8视频人员检测，GPU环境调试完成"