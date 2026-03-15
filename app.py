"""
Flask Web App for Video AI Upscaling (Real-ESRGAN + NVIDIA GPU)
- Upload a video, choose 4K or 8K, watch live progress
- Abort processing at any time
- Play input and output videos in the browser
"""

import os
import sys
import json
import time
import threading
import queue
import subprocess
import cv2
import torch
import imageio_ffmpeg
from flask import (
    Flask, render_template, request, jsonify, Response, send_from_directory,
)
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer

FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(SCRIPT_DIR, "input")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
MODEL_DIR = os.path.join(SCRIPT_DIR, "models")
ALLOWED_EXT = {".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm"}

os.makedirs(INPUT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

app = Flask(__name__, template_folder=os.path.join(SCRIPT_DIR, "templates"))

# ---------------------------------------------------------------------------
# Global state for the running job
# ---------------------------------------------------------------------------
job_lock = threading.Lock()
current_job = {
    "running": False,
    "abort": False,
    "filename": "",
    "progress": 0.0,
    "frame": 0,
    "total_frames": 0,
    "fps_speed": 0.0,
    "eta": "",
    "status": "idle",       # idle | processing | done | aborted | error
    "output_file": "",
}
progress_subscribers: list[queue.Queue] = []


def broadcast_progress():
    """Push current job state to all SSE subscribers."""
    data = json.dumps(current_job)
    dead = []
    for q in progress_subscribers:
        try:
            q.put_nowait(data)
        except queue.Full:
            dead.append(q)
    for q in dead:
        progress_subscribers.remove(q)


# ---------------------------------------------------------------------------
# Model helpers (reused from upscale_video.py)
# ---------------------------------------------------------------------------
MODEL_URLS = {
    "RealESRGAN_x2plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
    "RealESRGAN_x4plus": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
}


def download_model(model_name: str) -> str:
    model_path = os.path.join(MODEL_DIR, f"{model_name}.pth")
    if os.path.exists(model_path):
        return model_path
    url = MODEL_URLS[model_name]
    import urllib.request
    urllib.request.urlretrieve(url, model_path)
    return model_path


def create_upscaler(scale: int):
    gpu_id = 0 if torch.cuda.is_available() else None
    if scale <= 2:
        model_name = "RealESRGAN_x2plus"
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=2)
        netscale = 2
    else:
        model_name = "RealESRGAN_x4plus"
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                        num_block=23, num_grow_ch=32, scale=4)
        netscale = 4

    model_path = download_model(model_name)
    upscaler = RealESRGANer(
        scale=netscale, model_path=model_path, model=model,
        tile=400, tile_pad=10, pre_pad=0,
        half=gpu_id is not None, gpu_id=gpu_id,
    )
    return upscaler


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------
def has_audio_stream(filepath: str) -> bool:
    """Check if a video file contains an audio stream."""
    try:
        result = subprocess.run(
            [FFMPEG_BIN, "-i", filepath, "-hide_banner"],
            capture_output=True, text=True, timeout=15,
        )
        return "Audio:" in result.stderr
    except Exception:
        return False


def mux_audio(input_video: str, upscaled_video: str) -> bool:
    """Copy audio from input_video into upscaled_video. Returns True on success."""
    if not has_audio_stream(input_video):
        return False  # nothing to mux

    temp_path = upscaled_video + ".muxing.mp4"
    try:
        cmd = [
            FFMPEG_BIN, "-y",
            "-i", upscaled_video,   # upscaled video (no audio)
            "-i", input_video,      # original (has audio)
            "-c:v", "copy",         # keep upscaled video as-is
            "-c:a", "aac",          # re-encode audio to aac for mp4 compat
            "-map", "0:v:0",        # video from first input
            "-map", "1:a:0",        # audio from second input
            "-shortest",            # match shorter stream duration
            temp_path,
        ]
        subprocess.run(cmd, capture_output=True, timeout=600, check=True)
        # Replace original output with muxed version
        os.replace(temp_path, upscaled_video)
        return True
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False


# ---------------------------------------------------------------------------
# Background processing thread
# ---------------------------------------------------------------------------
def process_video(filename: str, scale: int):
    global current_job
    input_path = os.path.join(INPUT_DIR, filename)
    base, ext = os.path.splitext(filename)
    suffix = "4K" if scale == 2 else "8K"
    out_name = f"{base}_{suffix}{ext}"
    output_path = os.path.join(OUTPUT_DIR, out_name)

    current_job.update({
        "running": True, "abort": False, "filename": filename,
        "progress": 0.0, "frame": 0, "total_frames": 0,
        "fps_speed": 0.0, "eta": "", "status": "processing",
        "output_file": out_name,
    })
    broadcast_progress()

    try:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open {input_path}")

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        out_w, out_h = w * scale, h * scale

        current_job["total_frames"] = total
        broadcast_progress()

        upscaler = create_upscaler(scale)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
        if not writer.isOpened():
            raise RuntimeError("Cannot create output writer")

        frame_times = []
        for i in range(total):
            if current_job["abort"]:
                current_job["status"] = "aborted"
                break

            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.time()
            try:
                output, _ = upscaler.enhance(frame, outscale=scale)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                upscaler.tile = max(128, upscaler.tile // 2)
                output, _ = upscaler.enhance(frame, outscale=scale)

            if output.shape[1] != out_w or output.shape[0] != out_h:
                output = cv2.resize(output, (out_w, out_h),
                                    interpolation=cv2.INTER_LANCZOS4)
            writer.write(output)

            ft = time.time() - t0
            frame_times.append(ft)
            avg = sum(frame_times[-30:]) / len(frame_times[-30:])
            eta_s = avg * (total - i - 1)
            eta_str = f"{int(eta_s//3600):02d}:{int(eta_s%3600//60):02d}:{int(eta_s%60):02d}"

            current_job.update({
                "frame": i + 1,
                "progress": round((i + 1) / total * 100, 1),
                "fps_speed": round(ft, 2),
                "eta": eta_str,
            })
            broadcast_progress()

        cap.release()
        writer.release()

        if current_job["status"] == "aborted":
            # Clean up partial output
            if os.path.exists(output_path):
                os.remove(output_path)
        elif current_job["status"] == "processing":
            # Mux audio from original into upscaled output
            current_job["eta"] = "Syncing audio..."
            broadcast_progress()
            mux_audio(input_path, output_path)
            current_job["status"] = "done"
            current_job["progress"] = 100.0

    except Exception as e:
        current_job["status"] = "error"
        current_job["eta"] = str(e)
    finally:
        current_job["running"] = False
        broadcast_progress()
        # Free GPU memory
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Flask routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if current_job["running"]:
        return jsonify({"error": "A job is already running. Abort it first."}), 409

    file = request.files.get("video")
    if not file or file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"Unsupported format: {ext}"}), 400

    scale = int(request.form.get("scale", 2))
    if scale not in (2, 4):
        return jsonify({"error": "Scale must be 2 (4K) or 4 (8K)."}), 400

    # Save to input dir
    save_path = os.path.join(INPUT_DIR, file.filename)
    file.save(save_path)

    # Check if output already exists
    base, fext = os.path.splitext(file.filename)
    suffix = "4K" if scale == 2 else "8K"
    out_name = f"{base}_{suffix}{fext}"
    if os.path.exists(os.path.join(OUTPUT_DIR, out_name)):
        return jsonify({
            "status": "skipped",
            "message": f"Output already exists: {out_name}",
            "output_file": out_name,
            "input_file": file.filename,
        })

    # Start background processing
    t = threading.Thread(target=process_video, args=(file.filename, scale),
                         daemon=True)
    t.start()

    return jsonify({
        "status": "started",
        "message": f"Processing {file.filename} -> {out_name}",
        "input_file": file.filename,
        "output_file": out_name,
    })


@app.route("/abort", methods=["POST"])
def abort():
    if not current_job["running"]:
        return jsonify({"error": "No job running."}), 400
    current_job["abort"] = True
    return jsonify({"status": "aborting"})


@app.route("/status")
def status():
    return jsonify(current_job)


@app.route("/progress")
def progress_stream():
    """Server-Sent Events stream for live progress updates."""
    q = queue.Queue(maxsize=50)
    progress_subscribers.append(q)

    def stream():
        # Send current state immediately
        yield f"data: {json.dumps(current_job)}\n\n"
        try:
            while True:
                try:
                    data = q.get(timeout=30)
                    yield f"data: {data}\n\n"
                except queue.Empty:
                    # keepalive
                    yield f": keepalive\n\n"
        except GeneratorExit:
            pass
        finally:
            if q in progress_subscribers:
                progress_subscribers.remove(q)

    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/videos/input/<path:filename>")
def serve_input(filename):
    return send_from_directory(INPUT_DIR, filename)


@app.route("/videos/output/<path:filename>")
def serve_output(filename):
    return send_from_directory(OUTPUT_DIR, filename)


@app.route("/list")
def list_files():
    """List all input and output files with audio status."""
    inputs = sorted(f for f in os.listdir(INPUT_DIR)
                    if os.path.splitext(f)[1].lower() in ALLOWED_EXT)
    outputs = sorted(f for f in os.listdir(OUTPUT_DIR)
                     if os.path.splitext(f)[1].lower() in ALLOWED_EXT)
    # Check which outputs are missing audio
    outputs_no_audio = []
    for f in outputs:
        if not has_audio_stream(os.path.join(OUTPUT_DIR, f)):
            outputs_no_audio.append(f)
    return jsonify({
        "inputs": inputs,
        "outputs": outputs,
        "outputs_no_audio": outputs_no_audio,
    })


@app.route("/process", methods=["POST"])
def process_existing():
    """Start upscaling an existing input file (no re-upload needed)."""
    if current_job["running"]:
        return jsonify({"error": "A job is already running. Abort it first."}), 409

    data = request.get_json()
    filename = data.get("filename", "")
    scale = int(data.get("scale", 2))

    if scale not in (2, 4):
        return jsonify({"error": "Scale must be 2 (4K) or 4 (8K)."}), 400

    input_path = os.path.join(INPUT_DIR, filename)
    if not os.path.exists(input_path):
        return jsonify({"error": f"File not found: {filename}"}), 404

    base, fext = os.path.splitext(filename)
    suffix = "4K" if scale == 2 else "8K"
    out_name = f"{base}_{suffix}{fext}"
    if os.path.exists(os.path.join(OUTPUT_DIR, out_name)):
        return jsonify({
            "status": "skipped",
            "message": f"Output already exists: {out_name}",
            "output_file": out_name,
            "input_file": filename,
        })

    t = threading.Thread(target=process_video, args=(filename, scale),
                         daemon=True)
    t.start()

    return jsonify({
        "status": "started",
        "message": f"Processing {filename} -> {out_name}",
        "input_file": filename,
        "output_file": out_name,
    })


@app.route("/fix-audio", methods=["POST"])
def fix_audio():
    """Mux audio from the original input into an existing upscaled output."""
    data = request.get_json()
    input_file = data.get("input_file", "")
    output_file = data.get("output_file", "")

    input_path = os.path.join(INPUT_DIR, input_file)
    output_path = os.path.join(OUTPUT_DIR, output_file)

    if not os.path.exists(input_path):
        return jsonify({"error": f"Input not found: {input_file}"}), 404
    if not os.path.exists(output_path):
        return jsonify({"error": f"Output not found: {output_file}"}), 404

    if not has_audio_stream(input_path):
        return jsonify({"error": "Original video has no audio track."}), 400

    if has_audio_stream(output_path):
        return jsonify({"status": "skipped", "message": "Output already has audio."})

    success = mux_audio(input_path, output_path)
    if success:
        return jsonify({"status": "done", "message": f"Audio synced to {output_file}"})
    return jsonify({"error": "Audio mux failed."}), 500


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"\n  Input folder:  {INPUT_DIR}")
    print(f"  Output folder: {OUTPUT_DIR}")
    print(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'}")
    print(f"\n  Open http://localhost:7000 in your browser\n")
    app.run(host="0.0.0.0", port=7000, debug=False, threaded=True)
