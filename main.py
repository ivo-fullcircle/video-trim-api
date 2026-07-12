import os
import subprocess
import tempfile
import uuid

import requests
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(title="video-trim-api")

API_KEY = os.environ.get("API_KEY", "")


class TrimRequest(BaseModel):
    video_url: str
    start: float  # seconds
    end: float  # seconds


class ExtractAudioRequest(BaseModel):
    video_url: str


def check_api_key(x_api_key: str | None):
    if not API_KEY or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/trim")
def trim(req: TrimRequest, x_api_key: str | None = Header(default=None)):
    check_api_key(x_api_key)

    if req.end <= req.start:
        raise HTTPException(status_code=400, detail="end must be greater than start")

    duration = req.end - req.start
    if duration > 300:
        raise HTTPException(status_code=400, detail="clips longer than 5 minutes are not supported")

    job_id = uuid.uuid4().hex
    tmp_dir = tempfile.mkdtemp(prefix=f"trim-{job_id}-")
    input_path = os.path.join(tmp_dir, "input.mp4")
    output_path = os.path.join(tmp_dir, "output.mp4")

    try:
        _download(req.video_url, input_path)
        _run_ffmpeg_trim(input_path, output_path, req.start, duration)
        return FileResponse(output_path, media_type="video/mp4", filename="clip.mp4")
    finally:
        # FileResponse streams the file before this process exits normally,
        # so cleanup on the next request is handled by the OS tmp dir GC;
        # explicit cleanup here would delete the file before it's sent.
        pass


@app.post("/extract-audio")
def extract_audio(req: ExtractAudioRequest, x_api_key: str | None = Header(default=None)):
    check_api_key(x_api_key)

    job_id = uuid.uuid4().hex
    tmp_dir = tempfile.mkdtemp(prefix=f"audio-{job_id}-")
    input_path = os.path.join(tmp_dir, "input")
    output_path = os.path.join(tmp_dir, "output.mp3")

    _download(req.video_url, input_path)
    _run_ffmpeg_extract_audio(input_path, output_path)
    return FileResponse(output_path, media_type="audio/mpeg", filename="audio.mp3")


def _run_ffmpeg_extract_audio(input_path: str, output_path: str):
    # Low bitrate mono speech encoding: plenty for Whisper transcription,
    # keeps even long episodes comfortably under OpenAI's 25MB upload limit.
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "libmp3lame", "-b:a", "32k",
        output_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=280)
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="ffmpeg timed out after 280s")
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=f"ffmpeg failed: {result.stderr[-2000:]}")


def _download(url: str, dest_path: str):
    try:
        with requests.get(url, stream=True, timeout=180) as r:
            r.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    f.write(chunk)
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"could not download video_url: {e}")


def _run_ffmpeg_trim(input_path: str, output_path: str, start: float, duration: float):
    # Re-encode (not stream copy) so the cut lands exactly on start/duration
    # regardless of keyframe placement in the source file. Crop to 9:16 at the
    # source resolution FIRST, then scale to 1080x1920 — scaling first (as the
    # previous version did) blows up a landscape source to a huge intermediate
    # frame (e.g. 1920x1080 -> 3413x1920) before cropping, which timed out on
    # Render's free-tier CPU.
    crop = "crop='min(iw,ih*9/16)':'min(ih,iw*16/9)'"
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", input_path,
        "-t", str(duration),
        "-vf", f"{crop},scale=1080:1920",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        output_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=280)
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="ffmpeg timed out after 280s")
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=f"ffmpeg failed: {result.stderr[-2000:]}")
