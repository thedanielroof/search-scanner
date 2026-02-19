# SEARCH SCANNER — Claude Project Guide

## Overview
SEARCH SCANNER is a Matrix-themed Flask web app that runs locally on **port 8457**. It's a single-file application — all Python backend, HTML, CSS, and JavaScript live in one file: `image_finder.py`.

## File Locations
- **Running copy:** `~/.searchscanner/image_finder.py` (this is what the server runs)
- **Source copy:** `~/Documents/image_finder.py` (backup)
- **Git repo:** `~/Documents/search-scanner-repo/image_finder.py`
- **GitHub:** https://github.com/thedanielroof/search-scanner

**Always edit `~/.searchscanner/image_finder.py` first**, then sync to the other two locations.

## How to Sync, Restart, and Push
After any code change, always do all three:
```bash
# 1. Sync
cp ~/.searchscanner/image_finder.py ~/Documents/image_finder.py
cp ~/.searchscanner/image_finder.py ~/Documents/search-scanner-repo/image_finder.py

# 2. Restart (kill old process, start new)
lsof -ti:8457 | xargs kill -9 2>/dev/null
sleep 2
cd ~/.searchscanner && nohup python3 image_finder.py > /tmp/ss_server.log 2>&1 &

# 3. Push
cd ~/Documents/search-scanner-repo && git add image_finder.py && git commit -m "message" && git push
```

## App Architecture

### Single-File Structure (~6600 lines)
The file is organized in this order:
1. **Imports & Config** (lines 1–150) — Flask app, security, upload dirs, optional imports
2. **Helper Functions** (lines 150–350) — `_safe_path()`, `_add_scan_history()`, auth decorators
3. **Backend Classes:**
   - `ImageScanner` (~line 354) — perceptual hashing with Pillow/imagehash
   - `AudioScanner` (~line 570) — faster-whisper transcription + text search
   - `PLATFORM_SPECS` dict (~line 743) — social media platform specs (6 platforms)
   - `VideoAnalyzer` (~line 898) — OpenCV + PyAV video analysis with deep script/retention analysis
   - `WritingAnalyzer` (~line 2114) — 6-category writing scorer
4. **HTML_PAGE** (~line 2700–5560) — entire frontend as a Python string:
   - CSS with Matrix + Glass themes
   - Tab buttons: Image, Audio, Text, Algorithm, Writer, Transcribe
   - Panel HTML for each tab
   - All JavaScript (tab switching, uploads, progress polling, result rendering)
5. **LOGIN_PAGE** (~line 5560–5840) — login page HTML
6. **Flask Routes** (~line 5840–6660) — all API endpoints

### Tab System Pattern
Each tab follows the same pattern:
- **Button:** `<button onclick="switchMode('name')" id="tabName">`
- **Panel:** `<div class="mode-panel" id="panelName">`
- **JS function:** `switchMode(mode)` handles tab switching
- **Backend:** POST route starts background thread → GET route polls progress
- **History:** Saved via `_add_scan_history()` to `~/.searchscanner/scan_history.json`

### Themes
Two themes controlled by `applyTheme()`:
- **Matrix** (default) — green-on-black terminal aesthetic
- **Glass** — clean white/blue modern look
- CSS uses `body.glass` class overrides

## Tabs & What They Do

| Tab | Purpose | Backend Class | Key Routes |
|-----|---------|--------------|------------|
| **Image Scan** | Find visually similar images via perceptual hashing | `ImageScanner` | `/scan`, `/scan/progress` |
| **Audio Scan** | Transcribe audio/video + search text in transcripts | `AudioScanner` | `/audio/scan`, `/audio/scan/progress` |
| **Text Scan** | Search for text patterns in files | (inline) | `/scan` (with text mode) |
| **Algorithm** | Deep video analysis — platform scores, story, retention, engagement | `VideoAnalyzer` | `/algorithm/scan`, `/algorithm/scan/progress` |
| **Writer** | Writing quality analysis — 6 scoring categories | `WritingAnalyzer` | `/writer/analyze`, `/writer/analyze/progress` |
| **Transcribe** | Dedicated transcription with download | `AudioScanner` | `/audio/scan`, `/audio/scan/progress` |

## Dependencies
**Required:**
- `flask` — web framework
- `Pillow`, `imagehash` — image scanning
- `opencv-python-headless` (`cv2`) — video frame extraction, face detection

**Optional but expected:**
- `faster-whisper` — local Whisper transcription (free, runs locally)
- `av` (PyAV) — video/audio stream handling
- `python-docx` — .docx file reading (Writer tab)
- `PyPDF2` — .pdf file reading (Writer tab)

## Login
- Password: `1146` (hashed with SHA-256 + salt in code)
- Rate limiting: 5 attempts per 5 min, 10 min lockout
- Session timeout: 1 hour

## Key Conventions
- **Upload limit:** 3GB (`MAX_CONTENT_LENGTH`)
- **Use `python3`** not `python` (macOS)
- **Port conflicts:** Always kill port 8457 before restarting
- **Progress pattern:** All scans use background threading + progress dict + polling
- **Score grades:** A (85+), B (70-84), C (50-69), D (30-49), F (<30)
- **Score rendering:** `renderScoreRing(score, size)` JS function for SVG ring gauges
- **History:** `loadScanHistory(type)` with type-specific rendering branches
- **Escaping:** Always use `escapeHtml()` for user-provided text in innerHTML
