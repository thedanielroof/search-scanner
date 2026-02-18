# Search Scanner

Matrix-themed web app for scanning files by visual similarity, audio transcription, and text search. Built with Flask + Pillow + faster-whisper.

## Features

- **Image Scan** - Find visually similar images using perceptual hashing
- **Audio Scan** - Transcribe audio/video files with Whisper AI and search transcripts
- **Text Scan** - Search through text files in directories
- **Transcribe** - Full audio/video transcription with optional timestamps

## One-Click Deploy

### Deploy to Render (Free)

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/thedanielroof/search-scanner)

> After clicking, connect your GitHub fork of this repo. Set `SEARCHSCANNER_PASSWORD` to your desired login password.

### Deploy on Railway

[![Deploy on Railway](https://railway.com/button.svg)](https://railway.com/new/template?referralCode=&repo=https://github.com/thedanielroof/search-scanner)

> Connect your GitHub fork. Set `SEARCHSCANNER_PASSWORD` in the Variables tab.

## Local Setup

```bash
git clone https://github.com/thedanielroof/search-scanner.git
cd search-scanner
pip install -r requirements.txt
python image_finder.py
```

Open http://localhost:8457 - default password is `1146`.

## Docker

```bash
docker build -t search-scanner .
docker run -p 8457:8457 -e SEARCHSCANNER_PASSWORD=yourpassword search-scanner
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8457` | Server port (auto-set by cloud platforms) |
| `SEARCHSCANNER_PASSWORD` | `1146` | Login password |

## Cloud Notes

- **RAM**: Whisper tiny model uses ~200MB peak. Render free tier (512MB) works.
- **First audio scan**: Downloads the Whisper model (~75MB) automatically.
- **Storage**: Scan history resets on redeploy (ephemeral filesystem on free tiers).
