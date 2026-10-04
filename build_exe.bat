@echo off
chcp 936 >nul
cd /d "D:\NHKJ\find_p\video_person_capture"

echo ============================================
echo  第1步: 安装 PyInstaller
echo ============================================
.venv\Scripts\python.exe -m pip install pyinstaller -q

echo ============================================
echo  第2步: 开始打包 GPU 版 exe
echo ============================================
.venv\Scripts\python.exe -m PyInstaller --onedir --name person_capture --collect-all torch --collect-all torchvision --collect-all ultralytics --collect-all cv2 main.py

if errorlevel 1 (
  echo 打包失败，请查看上方错误信息
  pause
  exit /b 1
)

echo ============================================
echo  第3步: 复制配置和模型到 exe 目录
echo ============================================
copy /Y config.yaml dist\person_capture\ >nul
copy /Y yolov8n.pt dist\person_capture\ >nul
copy /Y yolov8x.pt dist\person_capture\ >nul

echo.
echo 打包完成! exe 位于: dist\person_capture\person_capture.exe
echo 运行 exe 看控制台是否打印 GPU 已启用
echo 如需隐藏黑窗, 在打包命令加 --noconsole 后重新打包
echo.
pause