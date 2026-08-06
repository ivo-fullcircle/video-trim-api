import json
import os
import subprocess
import tempfile
import uuid

import requests
import yt_dlp
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


class ProbeRequest(BaseModel):
    video_url: str


class FetchAudioRequest(BaseModel):
    source_url: str  # a page URL (YouTube/TikTok/Instagram/...), not a direct file link


class Caption(BaseModel):
    start: float
    end: float
    text: str


class ComposeRequest(BaseModel):
    background_video_url: str
    audio_url: str
    captions: list[Caption] = []


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


@app.post("/probe")
def probe(req: ProbeRequest, x_api_key: str | None = Header(default=None)):
    check_api_key(x_api_key)

    job_id = uuid.uuid4().hex
    tmp_dir = tempfile.mkdtemp(prefix=f"probe-{job_id}-")
    input_path = os.path.join(tmp_dir, "input")

    _download(req.video_url, input_path)
    return _run_ffprobe(input_path)


@app.post("/fetch-audio")
def fetch_audio(req: FetchAudioRequest, x_api_key: str | None = Header(default=None)):
    # For the reference-research engine: source_url is a YouTube/TikTok/Instagram
    # PAGE link (not a direct file), so the plain _download() used elsewhere can't
    # read it -- yt-dlp resolves the real media stream first. We only need audio
    # (for Whisper transcription), so we ask yt-dlp for audio-only to keep this
    # fast and avoid downloading full video we won't use.
    check_api_key(x_api_key)

    job_id = uuid.uuid4().hex
    tmp_dir = tempfile.mkdtemp(prefix=f"fetch-{job_id}-")
    output_template = os.path.join(tmp_dir, "audio.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "128",
        }],
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(req.source_url, download=True)
    except yt_dlp.utils.DownloadError as e:
        raise HTTPException(status_code=400, detail=f"could not fetch source_url: {e}")

    output_path = os.path.join(tmp_dir, "audio.mp3")
    if not os.path.exists(output_path):
        raise HTTPException(status_code=500, detail="yt-dlp did not produce an mp3 file")

    headers = {
        "X-Source-Title": (info.get("title") or "")[:200].encode("ascii", "ignore").decode(),
        "X-Source-Duration-Seconds": str(info.get("duration") or ""),
        "X-Source-Uploader": (info.get("uploader") or "")[:200].encode("ascii", "ignore").decode(),
    }
    return FileResponse(output_path, media_type="audio/mpeg", filename="audio.mp3", headers=headers)


def _run_ffprobe(input_path: str) -> dict:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height:format=duration",
        "-of", "json",
        input_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="ffprobe timed out after 60s")
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=f"ffprobe failed: {result.stderr[-2000:]}")

    data = json.loads(result.stdout)
    streams = data.get("streams", [])
    if not streams:
        raise HTTPException(status_code=400, detail="no video stream found in file")
    width = streams[0].get("width")
    height = streams[0].get("height")
    duration = float(data.get("format", {}).get("duration", 0))

    if width and height:
        orientation = "horizontal" if width > height else "vertical" if height > width else "square"
    else:
        orientation = "unknown"

    return {
        "width": width,
        "height": height,
        "duration_seconds": round(duration, 1),
        "orientation": orientation,
    }


@app.post("/compose")
def compose(req: ComposeRequest, x_api_key: str | None = Header(default=None)):
    check_api_key(x_api_key)

    job_id = uuid.uuid4().hex
    tmp_dir = tempfile.mkdtemp(prefix=f"compose-{job_id}-")
    bg_path = os.path.join(tmp_dir, "bg.mp4")
    audio_path = os.path.join(tmp_dir, "audio.mp3")
    srt_path = os.path.join(tmp_dir, "captions.srt")
    output_path = os.path.join(tmp_dir, "output.mp4")

    _download(req.background_video_url, bg_path)
    _download(req.audio_url, audio_path)
    audio_duration = _get_duration(audio_path)
    _write_srt(req.captions, srt_path)
    _run_ffmpeg_compose(bg_path, audio_path, srt_path, audio_duration, output_path)
    return FileResponse(output_path, media_type="video/mp4", filename="composed.mp4")


def _get_duration(path: str) -> float:
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", path]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=f"ffprobe failed: {result.stderr[-1000:]}")
    return float(json.loads(result.stdout)["format"]["duration"])


def _format_srt_time(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    hours, total_ms = divmod(total_ms, 3600000)
    minutes, total_ms = divmod(total_ms, 60000)
    secs, ms = divmod(total_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{ms:03d}"


def _write_srt(captions: list[Caption], path: str):
    with open(path, "w") as f:
        for i, c in enumerate(captions, start=1):
            f.write(f"{i}\n")
            f.write(f"{_format_srt_time(c.start)} --> {_format_srt_time(c.end)}\n")
            f.write(f"{c.text}\n\n")


def _run_ffmpeg_compose(bg_path: str, audio_path: str, srt_path: str, duration: float, output_path: str):
    # Loop the background to cover the full audio length, crop/scale to 1080x1920
    # (covers both landscape and already-vertical source loops), burn in captions
    # from the srt, and replace the background's own audio with the TTS track.
    srt_escaped = srt_path.replace("\\", "/").replace(":", "\\:")
    subtitle_style = (
        "FontSize=64,Bold=1,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
        "BorderStyle=1,Outline=3,Shadow=0,Alignment=2,MarginV=250"
    )
    cmd = [
        "ffmpeg", "-y",
        "-stream_loop", "-1", "-i", bg_path,
        "-i", audio_path,
        "-map", "0:v:0", "-map", "1:a:0",
        "-vf", f"scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,subtitles={srt_escaped}:force_style='{subtitle_style}'",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-t", str(duration),
        output_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=280)
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="ffmpeg timed out after 280s")
    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=f"ffmpeg failed: {result.stderr[-2000:]}")


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
