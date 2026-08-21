"""
硬解码探针脚本
验证本机 NVDEC 硬解码是否可用，对比硬解 vs 软解的解码帧率。

用法:
  python probe_hwdecode.py <视频文件路径>

输出:
  - 硬解码是否可用
  - 硬解码 fps（CUDA + 720P 输出）
  - 软解码 fps（OpenCV + 720P 缩放）
  - 加速比

注意事项:
  - 硬解码依赖 NVIDIA 驱动 + ffmpeg 编译了 --enable-cuda-nvdec
  - 如果硬解不可用，会自动降级测试软解并说明原因
  - 测试时长默认 10 秒，可通过 --duration 参数调整
"""

import sys
import time
import subprocess
import argparse
import numpy as np


def check_ffmpeg_hwaccel():
    """检查 ffmpeg 是否支持 cuda 硬件加速"""
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=10
        )
        # 检查是否有 h264_nvenc 编码器（间接验证 CUDA 可用）
        has_nvenc = "h264_nvenc" in result.stderr + result.stdout

        # 检查解码器
        result2 = subprocess.run(
            ["ffmpeg", "-hide_banner", "-decoders"],
            capture_output=True, text=True, timeout=10
        )
        has_nvdec = "h264_cuvid" in result2.stderr + result2.stdout

        return has_nvenc or has_nvdec, has_nvdec
    except FileNotFoundError:
        return False, False
    except Exception:
        return False, False


def probe_hwdecode(video_path: str, target_w: int = 1280, target_h: int = 720,
                   duration_sec: float = 10.0) -> dict:
    """
    用 ffmpeg -hwaccel cuda 管道解码，测帧率。

    Returns:
        dict: {fps, frame_count, elapsed, error}
    """
    frame_size = target_w * target_h * 3  # rawvideo bgr24

    cmd = [
        "ffmpeg",
        "-hide_banner", "-loglevel", "error",
        "-hwaccel", "cuda",
        "-i", video_path,
        "-vf", f"scale={target_w}:{target_h}",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "pipe:1",
    ]

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except FileNotFoundError:
        return {"fps": 0, "frame_count": 0, "elapsed": 0, "error": "ffmpeg 未找到"}

    frame_count = 0
    start = time.time()

    while time.time() - start < duration_sec:
        raw = proc.stdout.read(frame_size)
        if len(raw) < frame_size:
            break
        frame_count += 1

    elapsed = time.time() - start
    proc.kill()
    proc.wait()

    return {
        "fps": frame_count / elapsed if elapsed > 0 else 0,
        "frame_count": frame_count,
        "elapsed": elapsed,
        "error": None,
    }


def probe_softdecode(video_path: str, target_w: int = 1280, target_h: int = 720,
                     duration_sec: float = 10.0) -> dict:
    """
    用 OpenCV 软解码 + 缩放到 720P，测帧率。
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"fps": 0, "frame_count": 0, "elapsed": 0, "error": "无法打开视频"}

    frame_count = 0
    start = time.time()

    while time.time() - start < duration_sec:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]
        if w != target_w or h != target_h:
            frame = cv2.resize(frame, (target_w, target_h))
        frame_count += 1

    elapsed = time.time() - start
    cap.release()

    return {
        "fps": frame_count / elapsed if elapsed > 0 else 0,
        "frame_count": frame_count,
        "elapsed": elapsed,
        "error": None,
    }


def main():
    parser = argparse.ArgumentParser(description="硬解码探针：对比 NVDEC 硬解 vs OpenCV 软解")
    parser.add_argument("video", help="测试视频文件路径")
    parser.add_argument("--duration", type=float, default=10.0, help="测试时长（秒），默认10")
    parser.add_argument("--width", type=int, default=1280, help="输出宽度，默认1280(720P)")
    parser.add_argument("--height", type=int, default=720, help="输出高度，默认720")
    args = parser.parse_args()

    print("=" * 60)
    print("硬解码探针 (NVDEC vs OpenCV 软解)")
    print("=" * 60)
    print(f"视频: {args.video}")
    print(f"目标分辨率: {args.width}x{args.height}")
    print(f"测试时长: {args.duration}s")
    print()

    # 1. 检查 ffmpeg + CUDA 支持
    print("[1/3] 检查 ffmpeg CUDA 支持...")
    cuda_ok, nvdec_ok = check_ffmpeg_hwaccel()
    if cuda_ok:
        print(f"  ffmpeg 硬件加速: 可用 (NVDEC={'是' if nvdec_ok else '否'})")
    else:
        print("  ffmpeg 硬件加速: 不可用（未安装 ffmpeg 或未编译 CUDA 支持）")
        print("  → 跳过硬解码测试，仅测软解码")
    print()

    # 2. 硬解码测试
    hw_result = None
    if cuda_ok:
        print("[2/3] 硬解码测试 (ffmpeg -hwaccel cuda)...")
        hw_result = probe_hwdecode(args.video, args.width, args.height, args.duration)
        if hw_result["error"]:
            print(f"  错误: {hw_result['error']}")
        else:
            print(f"  解码帧数: {hw_result['frame_count']}")
            print(f"  耗时: {hw_result['elapsed']:.2f}s")
            print(f"  帧率: {hw_result['fps']:.1f} fps")
        print()

    # 3. 软解码测试
    print("[3/3] 软解码测试 (OpenCV)...")
    sw_result = probe_softdecode(args.video, args.width, args.height, args.duration)
    if sw_result["error"]:
        print(f"  错误: {sw_result['error']}")
    else:
        print(f"  解码帧数: {sw_result['frame_count']}")
        print(f"  耗时: {sw_result['elapsed']:.2f}s")
        print(f"  帧率: {sw_result['fps']:.1f} fps")
    print()

    # 4. 汇总
    print("=" * 60)
    print("汇总")
    print("=" * 60)
    if hw_result and hw_result["fps"] > 0 and sw_result["fps"] > 0:
        speedup = hw_result["fps"] / sw_result["fps"]
        print(f"  硬解码: {hw_result['fps']:.1f} fps")
        print(f"  软解码: {sw_result['fps']:.1f} fps")
        print(f"  加速比: {speedup:.2f}x")
        if speedup > 2.0:
            print("  结论: 硬解码显著加速，建议启用")
        elif speedup > 1.3:
            print("  结论: 硬解码有一定加速，可考虑启用")
        else:
            print("  结论: 硬解码加速不明显，软解码足够")
    elif sw_result["fps"] > 0:
        print(f"  软解码: {sw_result['fps']:.1f} fps")
        print("  硬解码: 不可用")
        print("  结论: 当前环境不支持硬解码，使用软解码")
    else:
        print("  两种方式均失败，请检查视频文件和依赖")


if __name__ == "__main__":
    main()
