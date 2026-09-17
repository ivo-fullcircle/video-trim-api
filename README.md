# video-trim-api

Small self-hosted service that trims a segment out of a video, given a public URL and a start/end time in seconds. Wraps `ffmpeg` (open source) behind a minimal REST API so n8n can call it over HTTP.

## Endpoints

- `GET /health` — returns `{"status": "ok"}`
- `POST /trim` — body `{"video_url": "...", "start": 12.5, "end": 42.0, "title_text": "...", "title_seconds": 1.4}`, header `X-API-Key: <secret>` — returns the trimmed MP4 file directly in the response body. `title_text` is optional: when set, burns an on-brand full-frame title card (the "graphic hook" -- what the video is about, shown before/over the talking head) into the first `title_seconds` (default 1.4s) of the clip.
- `POST /compose` — same optional `title_text`/`title_seconds` as `/trim`.
- `POST /title-card` — body `{"text": "...", "width": 1080, "height": 1920}` — returns just the title card as a PNG, for videos edited by hand (Filmora/CapCut): drop it as the first ~1-1.5s clip before the footage.

## Environment variables

- `API_KEY` — shared secret required in the `X-API-Key` header on every request.

## Deploy on Render

1. Push this repo to GitHub.
2. In Render: New → Web Service → connect this repo. Render detects the `Dockerfile` automatically.
3. Set the `API_KEY` environment variable to a random secret string.
4. Deploy. Render gives you a public URL like `https://video-trim-api.onrender.com`.

## Scaling notes

This is intentionally a synchronous, single-instance service meant for low-volume use (occasional short clips). If volume grows or multi-video concatenation is added, switch to an async submit/poll pattern (like Captions.ai's API) and move off the free tier before this becomes a bottleneck.
