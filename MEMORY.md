# SEARCH SCANNER — Development Memory

## Project History
Built iteratively through Claude conversations. Started as an image similarity finder, expanded into a multi-tool media analysis suite.

## Features Built (in order)
1. **Image Scan** — perceptual hash matching to find similar images across folders
2. **Audio Scan** — transcribe audio/video with Whisper, search transcripts for keywords
3. **Text Scan** — search for text patterns inside files in a directory
4. **Transcribe** — dedicated transcription tool with downloadable results
5. **Login Page** — password-protected with rate limiting and brute force protection
6. **Theme System** — Matrix (green terminal) + Glass (white modern) themes
7. **Algorithm** — deep video analysis for social media optimization:
   - Video properties (resolution, FPS, codec, duration, etc.) in 2 rows of 5
   - Platform scoring across 6 platforms (TikTok, YouTube, Instagram Reels, X/Twitter, LinkedIn, Facebook)
   - Deep story analysis with 10 dimensions
   - Retention analysis (re-hook density, drop-off risk, pacing variety)
   - Engagement timeline (segment-by-segment scoring)
   - 4 score gauges: Story, Retention, Emotional, Hook Strength
   - Strengths, delivery tips, story tips, retention tips
8. **Writer** — writing quality analyzer:
   - Paste text or upload files (.txt, .md, .docx, .pdf)
   - 6 scoring categories: Clarity, Engagement, Structure, Word Choice, Grammar & Style, Persuasion
   - Each category shows score ring, grade, strengths (green), improvements (amber)
   - Stats bar: word count, sentences, paragraphs, reading time
9. **Scan History** — persistent history for all scan types with type-specific rendering
10. **Live streaming results** — results appear as they're found during scans
11. **File upload limit** — raised to 3GB

## Things Removed
- Text Overlays video property (from Algorithm)
- Brightness video property (from Algorithm)
- Theme selector dropdown from both main page and login page
- "UPLOAD / SEARCH / MATCH" tagline

## Tab Order (current)
Image → Audio → Text → Algorithm → Writer → Transcribe

## Technical Decisions
- **Single-file architecture** — everything in one .py file for simplicity and portability
- **No database** — scan history stored as JSON file at `~/.searchscanner/scan_history.json`
- **Local-only AI** — Whisper runs locally (free), no API calls, no cloud dependencies
- **Background threading** — all scans run in threads with progress polling (not websockets)
- **Temp directories** — uploads go to OS temp dirs (auto-cleaned on reboot)
- **Two themes** — CSS variables with `body.glass` class override pattern

## Known Issues / Gotchas
- Use `python3` not `python` on macOS
- Must kill port 8457 before restart: `lsof -ti:8457 | xargs kill -9`
- `faster-whisper` and `av` must be installed for transcription to work (`HAS_WHISPER` flag)
- Video properties grid uses `repeat(5, 1fr)` — exactly 10 properties in 2 rows
- The `renderScoreRing()` JS function is shared between Algorithm and Writer tabs
- Glass theme requires explicit CSS overrides for every new UI component

## File Structure
```
~/.searchscanner/
├── image_finder.py          # The entire app (single file, ~6600 lines)
├── CLAUDE.md                # Claude project guide
├── MEMORY.md                # This file — development history
├── scan_history.json        # Persistent scan history (auto-created)
├── security.log             # Login attempt logs
└── secret_key.txt           # Flask session secret (auto-generated)
```

## GitHub
- Repo: https://github.com/thedanielroof/search-scanner
- Branch: main
- Always sync all 3 file copies before pushing

## Dependencies Install Command
```bash
pip3 install flask Pillow imagehash opencv-python-headless faster-whisper av python-docx PyPDF2
```
