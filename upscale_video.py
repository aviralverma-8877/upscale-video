"""
Batch Video Upscaler: Full HD -> 4K/8K with AI Enhancement
Uses Real-ESRGAN on NVIDIA GPU for high-quality super-resolution upscaling.

Requirements (install in order):
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
    pip install opencv-python numpy
    pip install basicsr realesrgan

Usage:
    python upscale_video.py
    - Place videos in the 'input' folder
    - Upscaled videos will be saved to the 'output' folder
    - Already processed files are automatically skipped
"""

import os
import sys
import time
import glob
import cv2
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "input")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm"}


def get_video_info(path: str) -> dict:
    """Read video metadata."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return info


def download_model(model_name: str) -> str:
    """Download the Real-ESRGAN model weights if not already cached."""
    model_dir = os.path.join(SCRIPT_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)

    model_urls = {
        "RealESRGAN_x2plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "RealESRGAN_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    }

    model_path = os.path.join(model_dir, f"{model_name}.pth")
    if os.path.exists(model_path):
        print(f"  Model already cached: {model_path}")
        return model_path

    url = model_urls.get(model_name)
    if url is None:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(model_urls.keys())}")

    print(f"  Downloading {model_name} model weights...")
    import urllib.request
    urllib.request.urlretrieve(url, model_path)
    print(f"  Model saved to: {model_path}")
    return model_path


def create_upscaler(scale: int, denoise_strength: float, tile_size: int) -> RealESRGANer:
    """Initialize the Real-ESRGAN upscaler with GPU acceleration."""
    if not torch.cuda.is_available():
        print("  WARNING: CUDA not available! Falling back to CPU (will be very slow).")
        gpu_id = None
    else:
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"  Using GPU: {gpu_name} ({vram:.1f} GB VRAM)")
        gpu_id = 0

    if scale <= 2:
        model_name = "RealESRGAN_x2plus"
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
    else:
        model_name = "RealESRGAN_x4plus"
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
        netscale = 4

    model_path = download_model(model_name)

    upscaler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=None if denoise_strength == 0 else [1 - denoise_strength, denoise_strength],
        model=model,
        tile=tile_size,
        tile_pad=10,
        pre_pad=0,
        half=True if gpu_id is not None else False,
        gpu_id=gpu_id,
    )
    return upscaler, netscale


def draw_progress_bar(progress: float, width: int = 40) -> str:
    """Return a text progress bar string."""
    filled = int(width * progress)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}] {progress * 100:5.1f}%"


def get_output_name(input_filename: str, scale: int) -> str:
    """Generate output filename with scale suffix."""
    base, ext = os.path.splitext(input_filename)
    suffix = "4K" if scale == 2 else "8K"
    return f"{base}_{suffix}{ext}"


def find_input_videos() -> list:
    """Find all video files in the input directory."""
    videos = []
    for f in sorted(os.listdir(INPUT_DIR)):
        ext = os.path.splitext(f)[1].lower()
        if ext in VIDEO_EXTENSIONS:
            videos.append(f)
    return videos


def upscale_video(
    input_path: str,
    output_path: str,
    upscaler: RealESRGANer,
    scale: int = 2,
    tile_size: int = 400,
):
    """Upscale a single video file."""
    info = get_video_info(input_path)
    out_w = info["width"] * scale
    out_h = info["height"] * scale

    print(f"  Resolution: {info['width']}x{info['height']} -> {out_w}x{out_h}")
    print(f"  FPS: {info['fps']} | Frames: {info['total_frames']}")

    cap = cv2.VideoCapture(input_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, info["fps"], (out_w, out_h))

    if not writer.isOpened():
        print("  ERROR: Cannot create output video writer.")
        cap.release()
        return False

    total = info["total_frames"]
    start_time = time.time()
    frame_times = []

    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        frame_start = time.time()

        try:
            output, _ = upscaler.enhance(frame, outscale=scale)
        except torch.cuda.OutOfMemoryError:
            print(f"\n  GPU OOM at frame {i+1}! Retrying with smaller tiles...")
            torch.cuda.empty_cache()
            upscaler.tile = max(128, upscaler.tile // 2)
            output, _ = upscaler.enhance(frame, outscale=scale)

        if output.shape[1] != out_w or output.shape[0] != out_h:
            output = cv2.resize(output, (out_w, out_h), interpolation=cv2.INTER_LANCZOS4)

        writer.write(output)

        frame_time = time.time() - frame_start
        frame_times.append(frame_time)
        avg_time = sum(frame_times[-30:]) / len(frame_times[-30:])
        eta = avg_time * (total - i - 1)

        progress = (i + 1) / total
        bar = draw_progress_bar(progress)
        eta_str = f"{int(eta//3600):02d}:{int(eta%3600//60):02d}:{int(eta%60):02d}"
        print(
            f"\r  {bar} | Frame {i+1}/{total} | "
            f"{frame_time:.2f}s/f | ETA: {eta_str}",
            end="", flush=True,
        )

    cap.release()
    writer.release()

    elapsed = time.time() - start_time
    in_size = os.path.getsize(input_path) / (1024 ** 2)
    out_size = os.path.getsize(output_path) / (1024 ** 2)
    print(f"\n  Done in {elapsed/60:.1f} min | Avg: {elapsed/max(total,1):.2f}s/f | "
          f"{in_size:.1f} MB -> {out_size:.1f} MB")
    return True


def ask_user_scale() -> int:
    """Ask the user to choose between 4K and 8K upscaling."""
    print("\n" + "=" * 60)
    print("  Video AI Upscaler (Real-ESRGAN + NVIDIA GPU)")
    print("=" * 60)
    print("\n  Select target resolution:\n")
    print("    [1] 4K  (3840x2160)  - 2x upscale  (recommended)")
    print("    [2] 8K  (7680x4320)  - 4x upscale  (slower, more VRAM)")

    while True:
        try:
            choice = input("\n  Enter your choice (1 or 2): ").strip()
            if choice == "1":
                print("  -> Selected: 4K upscaling (2x)")
                return 2
            elif choice == "2":
                print("  -> Selected: 8K upscaling (4x)")
                return 4
            else:
                print("  Invalid choice. Please enter 1 or 2.")
        except (KeyboardInterrupt, EOFError):
            print("\n  Cancelled.")
            sys.exit(0)


def main():
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find input videos
    videos = find_input_videos()

    if not videos:
        print(f"\nNo video files found in: {INPUT_DIR}")
        print(f"Supported formats: {', '.join(sorted(VIDEO_EXTENSIONS))}")
        print("Place your video files in the 'input' folder and run again.")
        sys.exit(0)

    print(f"\nFound {len(videos)} video(s) in input folder:")
    for v in videos:
        print(f"  - {v}")

    # Ask user for target resolution
    scale = ask_user_scale()
    suffix = "4K" if scale == 2 else "8K"

    # Determine which files to process vs skip
    to_process = []
    to_skip = []
    for v in videos:
        out_name = get_output_name(v, scale)
        out_path = os.path.join(OUTPUT_DIR, out_name)
        if os.path.exists(out_path):
            to_skip.append(v)
        else:
            to_process.append(v)

    print(f"\n  To process: {len(to_process)} | Already done (skip): {len(to_skip)}")

    # Show skipped files
    for v in to_skip:
        out_name = get_output_name(v, scale)
        print(f"\n  [{to_skip.index(v)+1}/{len(videos)}] {v}")
        print(f"  >> SKIPPED (output already exists: {out_name})")

    if not to_process:
        print("\nAll files already processed. Nothing to do.")
        return

    # Load model once for all videos
    denoise = 0.3
    tile_size = 400
    print(f"\n  Loading Real-ESRGAN model (scale={scale}x, denoise={denoise})...")
    upscaler, netscale = create_upscaler(scale, denoise, tile_size)
    print("  Model loaded successfully.")

    # Process each video
    file_num = len(to_skip)
    for idx, v in enumerate(to_process):
        file_num += 1
        input_path = os.path.join(INPUT_DIR, v)
        out_name = get_output_name(v, scale)
        output_path = os.path.join(OUTPUT_DIR, out_name)

        print(f"\n{'='*60}")
        print(f"  [{file_num}/{len(videos)}] Processing: {v} -> {out_name}")
        print(f"{'='*60}")

        success = upscale_video(
            input_path=input_path,
            output_path=output_path,
            upscaler=upscaler,
            scale=scale,
            tile_size=tile_size,
        )

        if not success:
            print(f"  FAILED: {v}")

    # Summary
    print(f"\n{'='*60}")
    print(f"  BATCH COMPLETE")
    print(f"{'='*60}")
    print(f"  Processed: {len(to_process)} file(s)")
    print(f"  Skipped:   {len(to_skip)} file(s)")
    print(f"  Output:    {OUTPUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
