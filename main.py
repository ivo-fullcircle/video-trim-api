import json
import os
import subprocess
import tempfile
import uuid

import requests
import yt_dlp
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel

app = FastAPI(title="video-trim-api")

API_KEY = os.environ.get("API_KEY", "")

# Title card ("hook" cover frame burned into the first ~1.4s of a clip) --
# brand colors from the Atendo design system (Navy + Teal). Font path points
# at fonts-dejavu-core, installed in the Dockerfile; override locally via
# env var when testing off a machine without that package installed.
TITLE_CARD_BG = (15, 23, 42)      # Navy #0F172A
TITLE_CARD_ACCENT = (20, 184, 166)  # Teal #14B8A6
TITLE_CARD_TEXT = (255, 255, 255)
TITLE_CARD_FONT_PATH = os.environ.get(
    "TITLE_CARD_FONT_PATH", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
)


class TrimRequest(BaseModel):
    video_url: str
    start: float  # seconds
    end: float  # seconds
    title_text: str | None = None  # optional hook/title card burned into the first title_seconds
    title_seconds: float = 1.4


class ExtractAudioRequest(BaseModel):
    video_url: str


class ProbeRequest(BaseModel):
    video_url: str


class FetchAudioRequest(BaseModel):
    source_url: str  # a page URL (YouTube/TikTok/Instagram/...), not a direct file link


class ListVideosRequest(BaseModel):
    profile_url: str  # a channel/profile page URL (YouTube channel, TikTok @user, ...)
    limit: int = 5


class Caption(BaseModel):
    start: float
    end: float
    text: str


class TitleCardRequest(BaseModel):
    text: str
    width: int = 1080
    height: int = 1920


class ComposeRequest(BaseModel):
    background_video_url: str
    audio_url: str
    captions: list[Caption] = []
    title_text: str | None = None  # optional hook/title card burned into the first title_seconds
    title_seconds: float = 1.4


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
        title_card_path = None
        if req.title_text:
            title_card_path = os.path.join(tmp_dir, "card.png")
            _render_title_card(req.title_text, 1080, 1920, title_card_path)
        _run_ffmpeg_trim(input_path, output_path, req.start, duration, title_card_path, req.title_seconds)
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


@app.post("/list-videos")
def list_videos(req: ListVideosRequest, x_api_key: str | None = Header(default=None)):
    # For the reference-research engine's auto-discovery: given a creator's
    # channel/profile page (not a specific video), list their most recent
    # videos WITHOUT downloading anything -- extract_flat skips resolving
    # each video's real media URL, so this is fast and cheap. The research
    # engine then picks one of these page URLs and feeds it to /fetch-audio
    # like it would any manually-pasted video link.
    check_api_key(x_api_key)

    profile_url = req.profile_url.rstrip("/")
    if "youtube.com/@" in profile_url and not any(
        profile_url.endswith(tab) for tab in ("/videos", "/shorts", "/streams", "/playlists")
    ):
        # A bare YouTube channel URL (.../@handle) resolves to the channel's
        # TABS (Videos/Live/Shorts) as flat entries, not actual videos --
        # the /videos tab is what actually lists individual uploads.
        profile_url = profile_url + "/videos"

    ydl_opts = {
        "extract_flat": "in_playlist",
        "playlistend": max(1, min(req.limit, 20)),
        "quiet": True,
        "no_warnings": True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(profile_url, download=False)
    except yt_dlp.utils.DownloadError as e:
        raise HTTPException(status_code=400, detail=f"could not list profile_url: {e}")

    entries = info.get("entries") if info else None
    if not entries:
        # Some extractors resolve a single-video URL straight to a video
        # entry instead of a playlist -- treat that as a 1-video result
        # rather than an error.
        if info and info.get("webpage_url"):
            entries = [info]
        else:
            raise HTTPException(status_code=404, detail="no videos found for this profile_url")

    videos = []
    for e in entries[: req.limit]:
        url = e.get("url") or e.get("webpage_url")
        if not url:
            continue
        videos.append({
            "url": url,
            "title": (e.get("title") or "")[:200],
            "duration_seconds": e.get("duration"),
        })

    return {"videos": videos}


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


@app.post("/title-card")
def title_card(req: TitleCardRequest, x_api_key: str | None = Header(default=None)):
    # Standalone title card as a plain PNG -- for videos edited by hand
    # (Filmora/CapCut), not just the automated /trim and /compose pipelines.
    # Drop this as the first ~1-1.5s clip before the talking-head footage.
    check_api_key(x_api_key)

    job_id = uuid.uuid4().hex
    tmp_dir = tempfile.mkdtemp(prefix=f"card-{job_id}-")
    out_path = os.path.join(tmp_dir, "card.png")
    _render_title_card(req.text, req.width, req.height, out_path)
    return FileResponse(out_path, media_type="image/png", filename="title-card.png")


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

    title_card_path = None
    if req.title_text:
        title_card_path = os.path.join(tmp_dir, "card.png")
        _render_title_card(req.title_text, 1080, 1920, title_card_path)

    _run_ffmpeg_compose(
        bg_path, audio_path, srt_path, audio_duration, output_path, title_card_path, req.title_seconds
    )
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


def _run_ffmpeg_compose(
    bg_path: str,
    audio_path: str,
    srt_path: str,
    duration: float,
    output_path: str,
    title_card_path: str | None = None,
    title_seconds: float = 1.4,
):
    # Loop the background to cover the full audio length, crop/scale to 1080x1920
    # (covers both landscape and already-vertical source loops), burn in captions
    # from the srt, and replace the background's own audio with the TTS track.
    srt_escaped = srt_path.replace("\\", "/").replace(":", "\\:")
    subtitle_style = (
        "FontSize=64,Bold=1,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
        "BorderStyle=1,Outline=3,Shadow=0,Alignment=2,MarginV=250"
    )
    base_vf = (
        f"scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920,"
        f"subtitles={srt_escaped}:force_style='{subtitle_style}'"
    )
    if title_card_path:
        # Third input (the pre-rendered title card PNG) is overlaid on top of the
        # composed frame only for the first title_seconds -- same single ffmpeg
        # pass, no extra encode, so it doesn't add runtime on Render's free tier.
        filter_complex = (
            f"[0:v]{base_vf}[base];"
            f"[2:v]format=rgba[card];"
            f"[base][card]overlay=0:0:enable='lt(t,{title_seconds})'[v]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-stream_loop", "-1", "-i", bg_path,
            "-i", audio_path,
            "-loop", "1", "-i", title_card_path,
            "-filter_complex", filter_complex,
            "-map", "[v]", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-t", str(duration),
            output_path,
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-stream_loop", "-1", "-i", bg_path,
            "-i", audio_path,
            "-map", "0:v:0", "-map", "1:a:0",
            "-vf", base_vf,
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


def _run_ffmpeg_trim(
    input_path: str,
    output_path: str,
    start: float,
    duration: float,
    title_card_path: str | None = None,
    title_seconds: float = 1.4,
):
    # Re-encode (not stream copy) so the cut lands exactly on start/duration
    # regardless of keyframe placement in the source file. Crop to 9:16 at the
    # source resolution FIRST, then scale to 1080x1920 — scaling first (as the
    # previous version did) blows up a landscape source to a huge intermediate
    # frame (e.g. 1920x1080 -> 3413x1920) before cropping, which timed out on
    # Render's free-tier CPU.
    crop = "crop='min(iw,ih*9/16)':'min(ih,iw*16/9)'"
    if title_card_path:
        # Second input (the pre-rendered title card PNG) is overlaid on top of
        # the cropped/scaled clip only for the first title_seconds -- same
        # single ffmpeg pass, no extra encode/runtime.
        filter_complex = (
            f"[0:v]{crop},scale=1080:1920[base];"
            f"[1:v]format=rgba[card];"
            f"[base][card]overlay=0:0:enable='lt(t,{title_seconds})'[v]"
        )
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", input_path,
            "-loop", "1", "-i", title_card_path,
            "-t", str(duration),
            "-filter_complex", filter_complex,
            "-map", "[v]", "-map", "0:a?",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            output_path,
        ]
    else:
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


def _wrap_text(draw: "ImageDraw.ImageDraw", text: str, font: "ImageFont.FreeTypeFont", max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        trial = f"{current} {word}".strip()
        if draw.textlength(trial, font=font) <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _render_title_card(text: str, width: int, height: int, out_path: str):
    # Full-frame cover card burned into the first ~1.4s of a video so viewers
    # know what it's about before the talking head or footage starts -- the
    # "graphic hook" every video in the Content System was missing. On-brand
    # Navy/Teal (Atendo design system), auto-wrapped and auto-shrunk to fit.
    img = Image.new("RGB", (width, height), TITLE_CARD_BG)
    draw = ImageDraw.Draw(img)

    max_text_width = int(width * 0.82)
    max_text_height = int(height * 0.42)
    font_size = 96
    lines: list[str] = [text]
    font = ImageFont.truetype(TITLE_CARD_FONT_PATH, font_size)
    while font_size > 36:
        font = ImageFont.truetype(TITLE_CARD_FONT_PATH, font_size)
        lines = _wrap_text(draw, text, font, max_text_width)
        line_height = font.getbbox("Ag")[3] + 18
        block_height = line_height * len(lines)
        widest = max((draw.textlength(line, font=font) for line in lines), default=0)
        if block_height <= max_text_height and widest <= max_text_width:
            break
        font_size -= 4

    line_height = font.getbbox("Ag")[3] + 18
    block_height = line_height * len(lines)

    # Teal accent bar centered above the text block, brand kicker style.
    bar_w, bar_h = 96, 8
    bar_x = (width - bar_w) // 2
    bar_y = height // 2 - block_height // 2 - 50
    draw.rectangle([bar_x, bar_y, bar_x + bar_w, bar_y + bar_h], fill=TITLE_CARD_ACCENT)

    y = height // 2 - block_height // 2
    for line in lines:
        line_w = draw.textlength(line, font=font)
        x = (width - line_w) // 2
        draw.text((x, y), line, font=font, fill=TITLE_CARD_TEXT)
        y += line_height

    img.save(out_path)
