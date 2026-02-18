#!/usr/bin/env python3
"""Image Similarity Finder - Web-based app to find visually similar images."""

import base64
import concurrent.futures
import hashlib
import json
import hmac
import io
import logging
import os
import secrets
import subprocess
import tempfile
import threading
import time
import uuid
from collections import defaultdict
from functools import wraps
from pathlib import Path

try:
    from PIL import Image
    import imagehash
except ImportError:
    print("Missing dependencies. Install with:")
    print("  pip3 install Pillow imagehash")
    raise SystemExit(1)

try:
    from flask import Flask, request, jsonify, send_file, session, redirect, url_for
except ImportError:
    print("Missing Flask. Install with:")
    print("  pip3 install flask")
    raise SystemExit(1)

# Optional: OpenCV for video frame scanning
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("Note: opencv-python not installed. Video scanning disabled.")
    print("  Install with: pip3 install opencv-python")

# Optional: faster-whisper for audio transcription
try:
    from faster_whisper import WhisperModel
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False
    print("Note: faster-whisper not installed. Audio scanning disabled.")
    print("  Install with: pip3 install faster-whisper")

import re
import numpy as np

# --- Constants ---
SUPPORTED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff', '.tif'}
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.m4v', '.flv', '.wmv'}
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.wma', '.aac', '.opus'}
WHISPER_MODEL_SIZE = 'tiny'
FRAME_SAMPLE_INTERVAL = 3.0  # Extract 1 frame per 3 seconds (faster, still catches scenes)
DEFAULT_THRESHOLD = 10
THUMBNAIL_SIZE = (150, 150)
MAX_RESULTS = 200

app = Flask(__name__)

# Persistent secret key — survives restarts so sessions stay valid
_SECRET_KEY_FILE = os.path.join(os.path.expanduser("~"), ".searchscanner", "secret.key")
os.makedirs(os.path.dirname(_SECRET_KEY_FILE), exist_ok=True)
if os.path.exists(_SECRET_KEY_FILE):
    with open(_SECRET_KEY_FILE, 'r') as f:
        app.secret_key = f.read().strip()
else:
    app.secret_key = secrets.token_hex(32)
    with open(_SECRET_KEY_FILE, 'w') as f:
        f.write(app.secret_key)
    os.chmod(_SECRET_KEY_FILE, 0o600)  # Owner-only read/write

# --- Security Configuration ---
# Password stored as salted SHA-256 hash (never plaintext)
_APP_SALT = 'S3archSc4nn3r_2026'
_APP_PASSWORD_HASH = hashlib.sha256((_APP_SALT + '1146').encode()).hexdigest()

# Session settings
app.config['SESSION_COOKIE_HTTPONLY'] = True    # JS can't access session cookie
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # CSRF protection for cookies
app.config['PERMANENT_SESSION_LIFETIME'] = 3600  # 1 hour session timeout
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max upload size

# Rate limiting / brute force protection
_login_attempts = defaultdict(list)  # IP -> [timestamps]
_locked_ips = {}                     # IP -> lockout_expiry_time
MAX_LOGIN_ATTEMPTS = 5               # Max failed attempts before lockout
LOGIN_WINDOW = 300                   # 5 minute window for attempt counting
LOCKOUT_DURATION = 600               # 10 minute lockout after max attempts

# CSRF token storage
_csrf_tokens = {}

# Security logger
_sec_log = logging.getLogger('security')
_SCANNER_HOME = os.path.join(os.path.expanduser("~"), ".searchscanner")
os.makedirs(_SCANNER_HOME, exist_ok=True)
_sec_handler = logging.FileHandler(os.path.join(_SCANNER_HOME, "security.log"))
_sec_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
_sec_log.addHandler(_sec_handler)
_sec_log.setLevel(logging.INFO)

UPLOAD_DIR = tempfile.mkdtemp(prefix="imgfinder_")
ALLOWED_UPLOAD_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp', '.tiff', '.tif'}

# Global scan progress tracking (thread-safe via lock)
_scan_progress = {}
_scan_lock = threading.Lock()

# Audio scan progress tracking
_audio_scan_progress = {}
_audio_scan_lock = threading.Lock()
AUDIO_UPLOAD_DIR = tempfile.mkdtemp(prefix="audioscan_")

# Lazy-loaded whisper model singleton
_whisper_model = None
_whisper_model_lock = threading.Lock()

def _get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        with _whisper_model_lock:
            if _whisper_model is None:
                _whisper_model = WhisperModel(
                    WHISPER_MODEL_SIZE,
                    device="cpu",
                    compute_type="int8",
                    num_workers=2,
                )
    return _whisper_model


# --- Security Helper Functions ---

def _verify_password(password):
    """Constant-time password comparison using HMAC to prevent timing attacks."""
    pw_hash = hashlib.sha256((_APP_SALT + password).encode()).hexdigest()
    return hmac.compare_digest(pw_hash, _APP_PASSWORD_HASH)


def _get_client_ip():
    """Get real client IP (handles proxies)."""
    return request.headers.get('X-Forwarded-For', request.remote_addr or '127.0.0.1').split(',')[0].strip()


def _is_ip_locked(ip):
    """Check if IP is currently locked out."""
    if ip in _locked_ips:
        if time.time() < _locked_ips[ip]:
            return True
        else:
            del _locked_ips[ip]
            _login_attempts.pop(ip, None)
    return False


def _record_failed_login(ip):
    """Record a failed login attempt. Returns True if IP is now locked."""
    now = time.time()
    _login_attempts[ip] = [t for t in _login_attempts[ip] if now - t < LOGIN_WINDOW]
    _login_attempts[ip].append(now)
    if len(_login_attempts[ip]) >= MAX_LOGIN_ATTEMPTS:
        _locked_ips[ip] = now + LOCKOUT_DURATION
        _sec_log.warning(f"IP LOCKED OUT: {ip} after {MAX_LOGIN_ATTEMPTS} failed attempts")
        return True
    return False


def _clear_login_attempts(ip):
    """Clear login attempts after successful login."""
    _login_attempts.pop(ip, None)
    _locked_ips.pop(ip, None)


def _generate_csrf_token():
    """Generate a CSRF token tied to the current session."""
    token = secrets.token_hex(32)
    sid = session.get('_id', secrets.token_hex(8))
    session['_id'] = sid
    _csrf_tokens[sid] = token
    return token


def _validate_csrf_token(token):
    """Validate CSRF token for the current session."""
    sid = session.get('_id')
    if not sid or sid not in _csrf_tokens:
        return False
    return hmac.compare_digest(_csrf_tokens[sid], token)


def _safe_path(path, must_be_under=None):
    """Sanitize and validate a file path against directory traversal attacks.
    Returns resolved absolute path or None if invalid."""
    if not path or not isinstance(path, str):
        return None
    # Block null bytes (classic path traversal trick)
    if '\x00' in path:
        _sec_log.warning(f"Null byte in path from {_get_client_ip()}: {repr(path[:50])}")
        return None
    # Expand and resolve to absolute (resolves .., symlinks)
    if path.startswith('~'):
        path = os.path.expanduser(path)
    resolved = os.path.realpath(os.path.abspath(path))
    # If must be under a specific directory, enforce it
    if must_be_under:
        must_be_under = os.path.realpath(os.path.abspath(must_be_under))
        if not resolved.startswith(must_be_under + os.sep) and resolved != must_be_under:
            _sec_log.warning(f"Path traversal blocked from {_get_client_ip()}: {repr(path[:80])} -> {resolved}")
            return None
    return resolved


def _validate_upload_file(file):
    """Validate uploaded file: extension, magic bytes, size."""
    if not file or not file.filename:
        return False, "No file provided"
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_UPLOAD_EXTENSIONS:
        _sec_log.warning(f"Blocked upload with extension {ext} from {_get_client_ip()}")
        return False, f"File type {ext} not allowed"
    # Read first 16 bytes to check magic bytes
    header = file.read(16)
    file.seek(0)
    image_signatures = [
        b'\x89PNG',         # PNG
        b'\xff\xd8\xff',    # JPEG
        b'GIF87a', b'GIF89a',  # GIF
        b'BM',              # BMP
        b'RIFF',            # WebP (RIFF container)
        b'II', b'MM',       # TIFF
    ]
    if not any(header.startswith(sig) for sig in image_signatures):
        _sec_log.warning(f"Blocked upload with invalid magic bytes from {_get_client_ip()}: {header[:8].hex()}")
        return False, "File does not appear to be a valid image"
    return True, None


def _validate_audio_upload(file):
    """Validate uploaded audio/video file: extension and magic bytes."""
    if not file or not file.filename:
        return False, "No file provided"
    ext = Path(file.filename).suffix.lower()
    all_media_exts = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS
    if ext not in all_media_exts:
        _sec_log.warning(f"Blocked audio upload with extension {ext} from {_get_client_ip()}")
        return False, f"File type {ext} not allowed"
    header = file.read(16)
    file.seek(0)
    audio_signatures = [
        b'\xff\xfb', b'\xff\xf3', b'\xff\xf2',  # MP3 frame sync
        b'ID3',            # MP3 with ID3 tag
        b'RIFF',           # WAV
        b'fLaC',           # FLAC
        b'OggS',           # OGG/Opus
    ]
    is_m4a_or_mp4 = b'ftyp' in header[:12]
    is_avi = header[:4] == b'RIFF' and header[8:12] == b'AVI '
    is_mkv = header[:4] == b'\x1a\x45\xdf\xa3'
    if not (any(header.startswith(sig) for sig in audio_signatures) or is_m4a_or_mp4 or is_avi or is_mkv):
        _sec_log.warning(f"Blocked audio upload with invalid magic bytes from {_get_client_ip()}: {header[:8].hex()}")
        return False, "File does not appear to be a valid audio/video file"
    return True, None


# Top-level function for ProcessPoolExecutor (must be picklable - can't be a method)
def _hash_image_file(path):
    """Hash a single image file using dhash with pre-resize. Returns (path, hash_str) or (path, None)."""
    try:
        with Image.open(path) as img:
            img.thumbnail((512, 512), Image.Resampling.LANCZOS)
            return path, str(imagehash.dhash(img))
    except Exception:
        return path, None
HISTORY_DIR = os.path.join(os.path.expanduser("~"), ".searchscanner", "history")
os.makedirs(HISTORY_DIR, exist_ok=True)

# Audio/Text/Transcribe scan history — stored as JSON entries (no large file copies)
SCAN_HISTORY_FILE = os.path.join(os.path.expanduser("~"), ".searchscanner", "scan_history.json")
_scan_history_lock = threading.Lock()

def _load_scan_history():
    try:
        with open(SCAN_HISTORY_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def _save_scan_history(entries):
    with open(SCAN_HISTORY_FILE, 'w') as f:
        json.dump(entries[-200:], f)  # Keep last 200 entries

def _add_scan_history(entry):
    """Add a scan result to history. Entry: {type, filename, date, duration, language, search_text, matches_count, segments_count, transcript_preview}"""
    with _scan_history_lock:
        entries = _load_scan_history()
        entry['id'] = uuid.uuid4().hex[:10]
        entry['date'] = time.strftime('%Y-%m-%d %H:%M:%S')
        entries.append(entry)
        _save_scan_history(entries)


class ImageScanner:
    def __init__(self, reference_path, search_dir, threshold=DEFAULT_THRESHOLD):
        self.reference_path = reference_path
        self.search_dir = search_dir
        self.threshold = threshold

    def compute_hash(self, image_path):
        try:
            with Image.open(image_path) as img:
                img.thumbnail((512, 512), Image.Resampling.LANCZOS)
                return imagehash.dhash(img)
        except Exception:
            return None

    def compute_video_best_match(self, video_path, ref_hash):
        """Extract frames from video and find the best matching frame.
        Optimized: sequential grab/skip (no seeking), grayscale hashing, dhash, thumbnail only at end."""
        if not HAS_CV2:
            return None
        cap = None
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return None
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps <= 0 or total_frames <= 0:
                return None
            frame_step = max(1, int(fps * FRAME_SAMPLE_INTERVAL))
            best_distance = self.threshold + 1
            best_timestamp = 0.0
            best_frame_raw = None  # Save raw frame only for best match
            frame_idx = 0
            next_hash_frame = 0
            while frame_idx < total_frames:
                if frame_idx == next_hash_frame:
                    # Decode this frame (need pixels for hashing)
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # Use grayscale for faster hashing
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    # Resize before hashing (much faster than full-res)
                    small = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_AREA)
                    pil_gray = Image.fromarray(small)
                    frame_hash = imagehash.dhash(pil_gray)
                    distance = ref_hash - frame_hash
                    if distance < best_distance:
                        best_distance = distance
                        best_timestamp = frame_idx / fps
                        # Save raw BGR frame bytes for thumbnail later (don't create PIL image yet)
                        best_frame_raw = frame
                        if distance == 0:
                            break  # Perfect match
                    next_hash_frame = frame_idx + frame_step
                else:
                    # Skip frame without decoding (much faster than read())
                    cap.grab()
                frame_idx += 1
            if best_distance <= self.threshold and best_frame_raw is not None:
                # Only now create the thumbnail PIL image for the best match
                rgb = cv2.cvtColor(best_frame_raw, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                pil_img.thumbnail((300, 300), Image.Resampling.LANCZOS)
                return (best_distance, best_timestamp, pil_img)
            return None
        except Exception:
            return None
        finally:
            if cap is not None:
                cap.release()

    def collect_media_paths(self):
        image_paths = []
        video_paths = []
        for root, _dirs, files in os.walk(self.search_dir, onerror=lambda e: None):
            try:
                for fname in files:
                    ext = Path(fname).suffix.lower()
                    full = os.path.join(root, fname)
                    if ext in SUPPORTED_EXTENSIONS:
                        image_paths.append(full)
                    elif ext in VIDEO_EXTENSIONS and HAS_CV2:
                        video_paths.append(full)
            except PermissionError:
                continue
        return image_paths, video_paths

    def scan(self, scan_id=None):
        ref_hash = self.compute_hash(self.reference_path)
        if ref_hash is None:
            raise ValueError(f"Could not read reference image: {self.reference_path}")
        image_paths, video_paths = self.collect_media_paths()
        total_images = len(image_paths)
        total_videos = len(video_paths)
        total_items = total_images + total_videos
        results = []
        errors = 0
        processed = 0
        ref_path_resolved = os.path.realpath(self.reference_path)

        def update_progress():
            """Update global progress dict for polling endpoint (merge, don't replace)."""
            if scan_id:
                with _scan_lock:
                    if scan_id in _scan_progress:
                        _scan_progress[scan_id].update({
                            'processed': processed,
                            'total': total_items,
                            'matches': len(results),
                            'total_images': total_images,
                            'total_videos': total_videos,
                        })

        update_progress()

        # Use all CPU cores — no artificial cap
        img_workers = os.cpu_count() or 4
        vid_workers = min(8, os.cpu_count() or 4)

        # ThreadPoolExecutor for both — ProcessPoolExecutor causes
        # "child process terminated abruptly" crashes on macOS due to fork issues
        # with Flask, OpenCV, and ObjC runtime. Pillow releases the GIL so threads work fine.
        img_executor = concurrent.futures.ThreadPoolExecutor(max_workers=img_workers)
        vid_executor = concurrent.futures.ThreadPoolExecutor(max_workers=vid_workers)

        try:
            # Submit ALL jobs at once — images + videos run concurrently
            img_future_map = {}
            for p in image_paths:
                f = img_executor.submit(_hash_image_file, p)
                img_future_map[f] = p

            vid_future_map = {}
            if video_paths:
                for vp in video_paths:
                    f = vid_executor.submit(self.compute_video_best_match, vp, ref_hash)
                    vid_future_map[f] = vp

            # Collect ALL results as they complete (images + videos interleaved)
            all_futures = list(img_future_map.keys()) + list(vid_future_map.keys())
            for future in concurrent.futures.as_completed(all_futures):
                if future in img_future_map:
                    path = img_future_map[future]
                    try:
                        _, hash_str = future.result()
                    except Exception:
                        errors += 1
                        processed += 1
                        update_progress()
                        continue
                    if hash_str is not None:
                        if os.path.realpath(path) == ref_path_resolved:
                            processed += 1
                            update_progress()
                            continue
                        candidate_hash = imagehash.hex_to_hash(hash_str)
                        distance = ref_hash - candidate_hash
                        if distance <= self.threshold:
                            results.append((path, distance, 'image', None))
                            # Stream partial result for live display
                            if scan_id:
                                try:
                                    thumb = make_thumbnail_b64(image_path=path)
                                    with _scan_lock:
                                        _scan_progress[scan_id]['partial_results'].append({
                                            'type': 'image', 'path': path,
                                            'folder': os.path.dirname(path),
                                            'distance': distance, 'thumbnail': thumb,
                                        })
                                except Exception:
                                    pass
                    else:
                        errors += 1
                    processed += 1
                    update_progress()
                elif future in vid_future_map:
                    vpath = vid_future_map[future]
                    try:
                        match = future.result()
                    except Exception:
                        errors += 1
                        processed += 1
                        update_progress()
                        continue
                    if match is not None:
                        dist, timestamp, frame_img = match
                        results.append((vpath, dist, 'video', {
                            'timestamp': timestamp,
                            'frame': frame_img,
                        }))
                        # Stream partial result for live display
                        if scan_id:
                            try:
                                thumb = make_thumbnail_b64(frame_image=frame_img)
                                with _scan_lock:
                                    _scan_progress[scan_id]['partial_results'].append({
                                        'type': 'video', 'path': vpath,
                                        'folder': os.path.dirname(vpath),
                                        'distance': dist, 'timestamp': timestamp,
                                        'thumbnail': thumb,
                                    })
                            except Exception:
                                pass
                    processed += 1
                    update_progress()
        finally:
            img_executor.shutdown(wait=False)
            vid_executor.shutdown(wait=False)

        results.sort(key=lambda r: r[1])
        if len(results) > MAX_RESULTS:
            results = results[:MAX_RESULTS]
        return results, errors, total_images, total_videos


class AudioScanner:
    """Transcribe audio and search transcript for keywords."""

    def __init__(self, audio_path, search_text):
        self.audio_path = audio_path
        self.search_text = search_text.strip().lower()

    def transcribe_and_search(self, scan_id=None, file_label=None, pct_base=0, pct_range=100):
        """Transcribe one audio file and search for keywords.
        pct_base/pct_range allow embedding within a larger multi-file progress bar."""
        model = _get_whisper_model()
        prefix = f'[{file_label}] ' if file_label else ''
        search_terms = [t.strip() for t in self.search_text.split() if t.strip()]

        def _update(phase, detail, pct):
            if not scan_id:
                return
            with _audio_scan_lock:
                if scan_id in _audio_scan_progress:
                    _audio_scan_progress[scan_id].update(
                        {'phase': phase, 'phase_detail': prefix + detail,
                         'percent': int(pct_base + pct_range * pct / 100)})

        def _push_partial(item):
            """Push a partial result for live streaming to the frontend."""
            if not scan_id:
                return
            with _audio_scan_lock:
                if scan_id in _audio_scan_progress:
                    _audio_scan_progress[scan_id]['partial_results'].append(item)

        _update('transcribing', 'Processing audio...', 5)

        segments_gen, info = model.transcribe(
            self.audio_path,
            beam_size=1,
            word_timestamps=False,
            language='en',
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=200,
            ),
        )

        full_transcript = []
        segment_list = []
        matches = []
        for seg in segments_gen:
            segment_list.append(seg)
            seg_data = {
                'id': len(segment_list) - 1,
                'start': seg.start,
                'end': seg.end,
                'text': seg.text.strip(),
            }
            full_transcript.append(seg_data)
            _update('transcribing', f'Transcribed {len(segment_list)} segments...', min(80, 5 + len(segment_list) * 2))

            # Stream each transcript segment live
            _push_partial({
                'kind': 'segment',
                'start': seg_data['start'],
                'end': seg_data['end'],
                'text': seg_data['text'],
            })

            # Check for keyword matches in real-time
            if search_terms:
                seg_lower = seg_data['text'].lower()
                matched = [t for t in search_terms if t in seg_lower]
                if matched:
                    highlighted = seg_data['text']
                    for term in matched:
                        pattern = re.compile(re.escape(term), re.IGNORECASE)
                        highlighted = pattern.sub(lambda m: f'<mark>{m.group()}</mark>', highlighted)
                    match_item = {
                        'segment_id': seg_data['id'],
                        'start': seg_data['start'],
                        'end': seg_data['end'],
                        'text': seg_data['text'],
                        'highlighted_text': highlighted,
                        'matched_terms': matched,
                    }
                    matches.append(match_item)
                    _push_partial({
                        'kind': 'match',
                        'start': match_item['start'],
                        'end': match_item['end'],
                        'highlighted_text': highlighted,
                        'matched_terms': matched,
                    })

        _update('done', 'Complete', 100)

        return {
            'matches': matches,
            'full_transcript': [{'start': s['start'], 'end': s['end'], 'text': s['text']} for s in full_transcript],
            'audio_duration': info.duration,
            'language': info.language,
            'language_probability': round(info.language_probability, 2) if info.language_probability else None,
            'total_segments': len(full_transcript),
        }

    @staticmethod
    def collect_audio_files(folder):
        """Walk a folder and return all audio + video files."""
        all_exts = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS
        files = []
        for root, dirs, filenames in os.walk(folder, onerror=lambda e: None):
            try:
                dirs[:] = [d for d in dirs if not d.startswith('.')]
            except PermissionError:
                continue
            for fname in sorted(filenames):
                if fname.startswith('.'):
                    continue
                ext = Path(fname).suffix.lower()
                if ext in all_exts:
                    files.append(os.path.join(root, fname))
        return files

    @staticmethod
    def extract_audio_from_video(video_path):
        """Extract audio track from a video file to a temp wav using PyAV (no ffmpeg CLI needed)."""
        try:
            import av as _av
            import wave
            temp_wav = os.path.join(AUDIO_UPLOAD_DIR, f"{uuid.uuid4().hex}.wav")
            container = _av.open(video_path)
            audio_stream = None
            for s in container.streams:
                if s.type == 'audio':
                    audio_stream = s
                    break
            if audio_stream is None:
                container.close()
                return None
            resampler = _av.AudioResampler(format='s16', layout='mono', rate=16000)
            with wave.open(temp_wav, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                for frame in container.decode(audio_stream):
                    resampled = resampler.resample(frame)
                    for r in resampled:
                        wf.writeframes(r.to_ndarray().tobytes())
            container.close()
            return temp_wav
        except Exception:
            return None


def make_thumbnail_b64(image_path=None, frame_image=None):
    try:
        if frame_image is not None:
            img = frame_image.copy()
        elif image_path is not None:
            img = Image.open(image_path)
        else:
            return None
        img.thumbnail(THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
        if img.mode in ('RGBA', 'P'):
            img = img.convert('RGB')
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=80)
        return base64.b64encode(buf.getvalue()).decode('ascii')
    except Exception:
        return None


HTML_PAGE = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SEARCH SCANNER</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Share Tech Mono', 'Courier New', monospace;
         background: #000a00; color: #00ff41; min-height: 100vh; position: relative; overflow-x: hidden; }
  #matrixCanvas { position: fixed; top: 0; left: 0; width: 100%; height: 100%;
                  z-index: 0; pointer-events: none; opacity: 0.12; }
  .app-wrap { position: relative; z-index: 1; }
  .app-wrap::before { content: ''; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    background: repeating-linear-gradient(0deg, rgba(0,0,0,0.15) 0px, rgba(0,0,0,0.15) 1px, transparent 1px, transparent 3px);
    pointer-events: none; z-index: 9999; }
  .header { background: rgba(0, 10, 0, 0.85); padding: 18px 30px;
            border-bottom: 2px solid #00ff41; display: flex; align-items: center; gap: 18px;
            justify-content: center; text-align: center;
            backdrop-filter: blur(6px); }
  .logo-svg { width: 56px; height: 56px; flex-shrink: 0; filter: drop-shadow(0 0 8px #00ff41); transition: filter 0.3s; }
  .logo-svg:hover { filter: drop-shadow(0 0 14px #00ff41) drop-shadow(0 0 30px #00ff4166); }
  /* Green eye cursor */
  * { cursor: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='28' height='18' viewBox='0 0 28 18'%3E%3Cpath d='M1 9 Q14 0 27 9 Q14 18 1 9 Z' fill='none' stroke='%2300ff41' stroke-width='1.5'/%3E%3Ccircle cx='14' cy='9' r='4' fill='none' stroke='%2300ff41' stroke-width='1.2'/%3E%3Ccircle cx='14' cy='9' r='1.5' fill='%2300ff41'/%3E%3C/svg%3E") 14 9, auto; }
  .header h1 { font-size: 22px; color: #00ff41; text-shadow: 0 0 10px #00ff41, 0 0 30px #008f11;
               letter-spacing: 2px; text-transform: uppercase; }
  .header p { color: #0a6e2a; font-size: 13px; letter-spacing: 1px; }
  .main { max-width: 1200px; margin: 0 auto; padding: 25px; }
  .controls { display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 25px; }
  .upload-zone { flex: 1; min-width: 280px; border: 2px dashed #0a6e2a;
                 border-radius: 4px; padding: 30px; text-align: center;
                 cursor: pointer; transition: all 0.3s; background: rgba(0, 15, 0, 0.7);
                 backdrop-filter: blur(4px); }
  .upload-zone:hover, .upload-zone.dragover { border-color: #00ff41; background: rgba(0, 30, 0, 0.8);
    box-shadow: 0 0 20px rgba(0, 255, 65, 0.15), inset 0 0 20px rgba(0, 255, 65, 0.05); }
  .upload-zone img { max-width: 200px; max-height: 200px; border-radius: 4px; margin-top: 10px;
                     border: 1px solid #0a6e2a; }
  .upload-zone.has-image { cursor: default; }
  .change-btn { display: inline-block; margin-top: 8px; padding: 5px 14px; background: transparent;
                color: #0a6e2a; border: 1px solid #0a3e1a; border-radius: 4px; cursor: pointer;
                font-size: 11px; font-family: 'Share Tech Mono', monospace; text-transform: uppercase;
                letter-spacing: 1px; transition: all 0.3s; }
  .change-btn:hover { border-color: #00ff41; color: #00ff41; }
  .upload-zone .upload-icon { margin-bottom: 10px; }
  .upload-zone .upload-icon svg { width: 60px; height: 60px; filter: drop-shadow(0 0 6px #00ff41); }
  .upload-zone p { color: #0a6e2a; margin: 5px 0; font-size: 13px; }
  .upload-zone p strong { color: #00ff41; }
  .settings { flex: 1; min-width: 280px; background: rgba(0, 15, 0, 0.7); border-radius: 4px;
              padding: 25px; border: 1px solid #0a6e2a; backdrop-filter: blur(4px); }
  .settings label { display: block; margin-bottom: 6px; font-weight: 600; color: #00ff41;
                    text-transform: uppercase; font-size: 12px; letter-spacing: 1px; }
  .settings input[type="text"] { width: 100%; padding: 10px 12px; border-radius: 4px;
                                  border: 1px solid #0a6e2a; background: rgba(0, 5, 0, 0.9);
                                  color: #00ff41; font-size: 14px; margin-bottom: 15px;
                                  font-family: 'Share Tech Mono', monospace; }
  .settings input[type="text"]:focus { outline: none; border-color: #00ff41;
    box-shadow: 0 0 10px rgba(0, 255, 65, 0.3); }
  .settings input[type="text"]::placeholder { color: #0a4e1a; }
  .folder-row { display: flex; gap: 8px; margin-bottom: 15px; }
  .folder-row input { flex: 1; }
  .browse-btn { padding: 10px 18px; background: transparent; color: #00ff41; border: 1px solid #00ff41;
                border-radius: 4px; cursor: pointer; font-size: 13px; white-space: nowrap;
                font-family: 'Share Tech Mono', monospace; text-transform: uppercase;
                letter-spacing: 1px; transition: all 0.3s; }
  .browse-btn:hover { background: #00ff41; color: #000a00;
    box-shadow: 0 0 12px rgba(0, 255, 65, 0.5); }
  .scan-btn { width: 100%; padding: 14px; background: transparent; color: #00ff41;
              border: 2px solid #00ff41; border-radius: 4px; font-size: 15px; font-weight: 600;
              cursor: pointer; transition: all 0.3s; text-transform: uppercase; letter-spacing: 3px;
              font-family: 'Share Tech Mono', monospace; }
  .scan-btn:hover { background: #00ff41; color: #000a00;
    box-shadow: 0 0 20px #00ff41, 0 0 40px rgba(0, 255, 65, 0.3); }
  .scan-btn:disabled { border-color: #0a3e1a; color: #0a3e1a; background: transparent;
                       cursor: not-allowed; box-shadow: none; }
  .status { text-align: center; padding: 15px; color: #0a6e2a; font-size: 14px; }
  .status .spinner { display: inline-block; width: 20px; height: 20px;
                     border: 3px solid #0a3e1a; border-top-color: #00ff41;
                     border-radius: 50%; animation: spin 0.8s linear infinite;
                     vertical-align: middle; margin-right: 8px; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .progress-bar { width: 100%; height: 10px; background: #0a3e1a; border-radius: 5px;
                  margin-top: 10px; overflow: hidden; border: 1px solid #0a5e2a; }
  .progress-bar .fill { height: 100%; background: linear-gradient(90deg, #00ff41, #7fff00);
                        transition: width 0.3s ease; border-radius: 5px;
                        box-shadow: 0 0 12px #00ff41, 0 0 4px #00ff41 inset; }
  .results { display: flex; flex-direction: column; gap: 12px; }
  .results-header { font-size: 16px; font-weight: 600; margin-bottom: 5px; color: #00ff41;
                    text-transform: uppercase; letter-spacing: 1px; text-shadow: 0 0 6px #00ff41; }
  .result-card { display: flex; gap: 16px; background: rgba(0, 15, 0, 0.75); border-radius: 4px;
                 padding: 14px; border: 1px solid #0a3e1a; transition: all 0.3s;
                 align-items: center; backdrop-filter: blur(4px); }
  .result-card:hover { border-color: #00ff41; box-shadow: 0 0 15px rgba(0, 255, 65, 0.15); }
  .result-card img { width: 120px; height: 120px; object-fit: cover; border-radius: 4px;
                     flex-shrink: 0; background: #000; border: 1px solid #0a3e1a; }
  .result-info { flex: 1; min-width: 0; }
  .result-info .similarity { font-size: 15px; font-weight: 700; margin-bottom: 4px; }
  .result-info .similarity.high { color: #00ff41; text-shadow: 0 0 8px #00ff41; }
  .result-info .similarity.med { color: #7fff00; text-shadow: 0 0 6px #7fff00; }
  .result-info .similarity.low { color: #ff6e41; text-shadow: 0 0 6px #ff6e41; }
  .result-info .filepath { color: #0a8e3a; font-size: 12px; word-break: break-all; margin-bottom: 4px; }
  .result-info .folder { color: #0a5e2a; font-size: 11px; word-break: break-all; }
  .reveal-btn { padding: 8px 16px; background: transparent; color: #00ff41;
                border: 1px solid #00ff41; border-radius: 4px; cursor: pointer;
                font-size: 12px; white-space: nowrap; flex-shrink: 0;
                font-family: 'Share Tech Mono', monospace; text-transform: uppercase;
                letter-spacing: 1px; transition: all 0.3s; }
  .reveal-btn:hover { background: #00ff41; color: #000a00;
    box-shadow: 0 0 12px rgba(0, 255, 65, 0.5); }
  .empty-state { text-align: center; padding: 60px 20px; color: #0a4e2a; }
  .empty-state .icon { font-size: 64px; margin-bottom: 15px; filter: drop-shadow(0 0 6px #0a6e2a); }
  .empty-state p { font-size: 14px; }
  .history-section { margin-bottom: 25px; }
  .history-toggle { background: transparent; color: #0a6e2a; border: 1px solid #0a3e1a;
                    border-radius: 4px; padding: 8px 16px; cursor: pointer; font-size: 13px;
                    font-family: 'Share Tech Mono', monospace; text-transform: uppercase;
                    letter-spacing: 1px; transition: all 0.3s; width: 100%; text-align: left; }
  .history-toggle:hover { border-color: #00ff41; color: #00ff41; }
  .history-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
                  gap: 10px; margin-top: 12px; padding: 12px; background: rgba(0, 10, 0, 0.5);
                  border: 1px solid #0a3e1a; border-radius: 4px; }
  .history-item { cursor: pointer; border: 2px solid transparent; border-radius: 4px;
                  overflow: hidden; transition: all 0.3s; position: relative; }
  .history-item:hover { border-color: #00ff41; box-shadow: 0 0 10px rgba(0, 255, 65, 0.3); }
  .history-item img { width: 100%; height: 80px; object-fit: cover; display: block; }
  .history-item .hist-name { font-size: 9px; color: #0a6e2a; padding: 3px 4px;
                             white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
                             background: rgba(0, 5, 0, 0.8); }
  .history-item .hist-delete { position: absolute; top: 3px; right: 3px; width: 18px; height: 18px;
    background: rgba(0,0,0,0.7); border: 1px solid #ff6e41; border-radius: 50%; color: #ff6e41;
    font-size: 11px; line-height: 16px; text-align: center; cursor: pointer; display: none;
    font-family: 'Share Tech Mono', monospace; transition: all 0.2s; z-index: 2; }
  .history-item .hist-delete:hover { background: #ff6e41; color: #000; }
  .history-item:hover .hist-delete { display: block; }
  .history-path { font-size: 11px; color: #0a5e2a; margin-top: 6px; }
  .history-controls { display: flex; gap: 10px; align-items: center; margin-top: 8px; margin-bottom: 4px; }
  .history-clear-btn { background: transparent; color: #ff6e41; border: 1px solid #ff6e41;
    border-radius: 4px; padding: 5px 12px; cursor: pointer; font-size: 11px;
    font-family: 'Share Tech Mono', monospace; text-transform: uppercase;
    letter-spacing: 1px; transition: all 0.3s; }
  .history-clear-btn:hover { background: #ff6e41; color: #000; }
  .scan-history-item { padding: 10px 12px; border: 1px solid #0a3e1a; border-radius: 4px;
    margin-bottom: 8px; background: rgba(0, 10, 0, 0.3); transition: border-color 0.3s; }
  .scan-history-item:hover { border-color: #00ff41; }
  .video-badge { display: inline-block; background: #ff6e41; color: #000; padding: 2px 8px;
    border-radius: 3px; font-size: 10px; font-weight: 700; letter-spacing: 1px; margin-right: 8px;
    text-shadow: none; vertical-align: middle; }
  .timestamp { color: #7fff00; font-size: 11px; margin-top: 4px; }

  @keyframes flicker { 0%, 95% { opacity: 1; } 96% { opacity: 0.8; } 97% { opacity: 1; }
    98% { opacity: 0.6; } 100% { opacity: 1; } }
  .header h1 { animation: flicker 4s infinite; }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

  /* Folder browser modal */
  .modal-overlay { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    background: rgba(0,0,0,0.85); z-index: 10000; justify-content: center; align-items: center; }
  .modal-overlay.open { display: flex; }
  .folder-modal { background: #000a00; border: 2px solid #00ff41; border-radius: 8px;
    width: 550px; max-width: 90vw; max-height: 80vh; display: flex; flex-direction: column;
    box-shadow: 0 0 40px rgba(0,255,65,0.2); }
  .folder-modal-header { padding: 14px 18px; border-bottom: 1px solid #0a3e1a;
    display: flex; justify-content: space-between; align-items: center; }
  .folder-modal-header h2 { font-size: 15px; color: #00ff41; letter-spacing: 1px; }
  .modal-close { background: transparent; border: 1px solid #0a3e1a; color: #0a6e2a;
    padding: 4px 12px; cursor: pointer; font-family: 'Share Tech Mono', monospace;
    font-size: 12px; border-radius: 4px; transition: all 0.3s; }
  .modal-close:hover { border-color: #ff6e41; color: #ff6e41; }
  .folder-modal-path { padding: 10px 18px; border-bottom: 1px solid #0a3e1a;
    font-size: 12px; color: #0a8e3a; word-break: break-all; background: rgba(0,15,0,0.5); }
  .folder-modal-list { flex: 1; overflow-y: auto; padding: 8px 0; min-height: 200px; max-height: 50vh; }
  .folder-modal-list::-webkit-scrollbar { width: 8px; }
  .folder-modal-list::-webkit-scrollbar-track { background: #000a00; }
  .folder-modal-list::-webkit-scrollbar-thumb { background: #0a3e1a; border-radius: 4px; }
  .folder-item { padding: 8px 18px; cursor: pointer; display: flex; align-items: center;
    gap: 10px; font-size: 13px; color: #00ff41; transition: background 0.15s; }
  .folder-item:hover { background: rgba(0,255,65,0.08); }
  .folder-item .icon { font-size: 16px; flex-shrink: 0; width: 20px; text-align: center; }
  .folder-item.parent { color: #0a8e3a; }
  .folder-modal-footer { padding: 12px 18px; border-top: 1px solid #0a3e1a;
    display: flex; justify-content: flex-end; gap: 10px; }
  .modal-select-btn { padding: 10px 24px; background: transparent; color: #00ff41;
    border: 2px solid #00ff41; border-radius: 4px; cursor: pointer; font-size: 13px;
    font-family: 'Share Tech Mono', monospace; text-transform: uppercase;
    letter-spacing: 1px; transition: all 0.3s; }
  .modal-select-btn:hover { background: #00ff41; color: #000a00;
    box-shadow: 0 0 12px rgba(0,255,65,0.5); }

  /* Mode tabs */
  .mode-tabs { display:flex; gap:0; margin-bottom:25px; border-bottom:2px solid #0a3e1a; }
  .mode-tab { flex:1; padding:12px 20px; background:transparent; color:#0a6e2a;
    border:none; border-bottom:2px solid transparent; cursor:pointer;
    font-family:'Share Tech Mono',monospace; font-size:14px;
    text-transform:uppercase; letter-spacing:2px; transition:all 0.3s; margin-bottom:-2px; }
  .mode-tab:hover { color:#00ff41; }
  .mode-tab.active { color:#00ff41; border-bottom-color:#00ff41; text-shadow:0 0 8px #00ff41; }
  .mode-panel { display:none; }
  .mode-panel.active { display:block; }

  /* Audio scan styles */
  .search-text-input { width:100%; padding:10px 12px; border-radius:4px;
    border:1px solid #0a6e2a; background:rgba(0,5,0,0.9);
    color:#00ff41; font-size:14px; margin-bottom:15px;
    font-family:'Share Tech Mono',monospace; }
  .search-text-input:focus { outline:none; border-color:#00ff41;
    box-shadow:0 0 10px rgba(0,255,65,0.3); }
  .search-text-input::placeholder { color:#0a4e1a; }
  .audio-badge { display:inline-block; background:#41a0ff; color:#000;
    padding:2px 8px; border-radius:3px; font-size:10px; font-weight:700;
    letter-spacing:1px; margin-right:8px; }
  .transcript-segment { background:rgba(0,15,0,0.75); border:1px solid #0a3e1a;
    border-radius:4px; padding:14px; margin-bottom:8px; transition:all 0.3s; }
  .transcript-segment:hover { border-color:#00ff41; box-shadow:0 0 15px rgba(0,255,65,0.15); }
  .transcript-segment mark { background:rgba(0,255,65,0.25); color:#00ff41;
    padding:1px 3px; border-radius:2px; font-weight:700; }
  .transcript-segment .seg-time { color:#7fff00; font-size:12px; font-weight:600; margin-bottom:6px; }
  .transcript-segment .seg-text { color:#0a8e3a; font-size:14px; line-height:1.6; }
  .transcript-segment .matched-terms { font-size:11px; color:#0a5e2a; margin-top:6px; }
  .full-transcript { margin-top:10px; border:1px solid #0a3e1a; border-radius:4px;
    padding:16px; background:rgba(0,10,0,0.5); max-height:400px;
    overflow-y:auto; font-size:13px; line-height:1.8; color:#0a6e2a; }
</style>
</head>
<body>

<canvas id="matrixCanvas"></canvas>

<div class="app-wrap">
<div class="header">
  <svg class="logo-svg" id="logoSvg" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
    <circle cx="40" cy="40" r="28" fill="none" stroke="#00ff41" stroke-width="4"/>
    <line x1="60" y1="60" x2="88" y2="88" stroke="#00ff41" stroke-width="6" stroke-linecap="round"/>
    <path d="M22 40 Q40 25 58 40 Q40 55 22 40 Z" fill="none" stroke="#00ff41" stroke-width="2.5"/>
    <circle id="eyeIris" cx="40" cy="40" r="7" fill="none" stroke="#00ff41" stroke-width="2"/>
    <circle id="eyePupil" cx="40" cy="40" r="3" fill="#00ff41"/>
    <circle id="eyeGlint" cx="37" cy="37" r="1.5" fill="#aaffaa"/>
  </svg>
  <div>
    <h1>SEARCH SCANNER</h1>
    <p>UPLOAD / SEARCH / MATCH</p>
  </div>
</div>

<div class="main">
  <div class="mode-tabs top-mode-tabs">
    <button class="mode-tab active" onclick="switchMode('image')" id="tabImage"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:4px"><rect x="2" y="3" width="20" height="18" rx="2"/><circle cx="8.5" cy="10.5" r="2.5"/><path d="M21 15l-5-5L5 21"/></svg>Image</button>
    <button class="mode-tab" onclick="switchMode('audio')" id="tabAudio"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:4px"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.54 8.46a5 5 0 010 7.07"/><path d="M19.07 4.93a10 10 0 010 14.14"/></svg>Audio</button>
    <button class="mode-tab" onclick="switchMode('text')" id="tabText"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:4px"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>Text</button>
    <button class="mode-tab" onclick="switchMode('transcribe')" id="tabTranscribe"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:4px"><path d="M17 3a2.828 2.828 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5L17 3z"/><path d="M15 5l4 4"/></svg>Transcribe</button>
  </div>

  <div class="mode-panel active" id="panelImage">
  <div class="controls">
    <div class="upload-zone" id="dropZone" onclick="if(!uploadedFile) document.getElementById('fileInput').click()">
      <div class="upload-icon">
        <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
          <circle cx="40" cy="40" r="28" fill="none" stroke="#00ff41" stroke-width="3" opacity="0.5"/>
          <line x1="60" y1="60" x2="85" y2="85" stroke="#00ff41" stroke-width="5" stroke-linecap="round" opacity="0.5"/>
          <path d="M22 40 Q40 27 58 40 Q40 53 22 40 Z" fill="none" stroke="#00ff41" stroke-width="2" opacity="0.5"/>
          <circle cx="40" cy="40" r="6" fill="none" stroke="#00ff41" stroke-width="1.5" opacity="0.5"/>
          <circle cx="40" cy="40" r="2.5" fill="#00ff41" opacity="0.5"/>
        </svg>
      </div>
      <p><strong>[ Click or drag & drop ]</strong></p>
      <p>upload image to search</p>
      <input type="file" id="fileInput" accept="image/*" hidden>
      <img id="preview" hidden>
    </div>

    <div class="settings">
      <label>> Search Folder</label>
      <div class="folder-row">
        <input type="text" id="folderPath" placeholder="Type path or click Browse...">
        <button class="browse-btn" onclick="openBrowseModal('folderPath')">Browse</button>
      </div>
      <button class="scan-btn" id="scanBtn" onclick="startScan()" disabled>[ Initiate Scan ]</button>
    </div>
  </div>

  <div class="history-section">
    <button class="history-toggle" id="historyToggle" onclick="toggleHistory()">
      > Upload History [ Click to expand ]
    </button>
    <div id="historyPanel" hidden>
      <p class="history-path" id="historyPath"></p>
      <div class="history-controls" id="historyControls" hidden>
        <button class="history-clear-btn" onclick="clearAllHistory()">[ Clear All History ]</button>
        <span style="color:#0a5e2a;font-size:11px;" id="historyCount"></span>
      </div>
      <div class="history-grid" id="historyGrid">
        <p style="color:#0a4e2a;grid-column:1/-1;text-align:center;">Loading...</p>
      </div>
    </div>
  </div>

  <div class="status" id="status" hidden>
    <div style="display:flex;align-items:center;justify-content:center;gap:10px;">
      <span class="spinner"></span> <span id="statusText">Scanning...</span>
    </div>
    <div id="timerDisplay" style="text-align:center;margin-top:6px;font-size:20px;color:#00ff41;text-shadow:0 0 10px #00ff41;letter-spacing:2px;">0:00.0</div>
    <div class="progress-bar"><div class="fill" id="progressFill" style="width:0%"></div></div>
  </div>

  <div id="results">
    <div class="empty-state">
      <div class="icon">
        <svg width="80" height="80" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
          <circle cx="40" cy="40" r="28" fill="none" stroke="#0a4e2a" stroke-width="3"/>
          <line x1="60" y1="60" x2="85" y2="85" stroke="#0a4e2a" stroke-width="5" stroke-linecap="round"/>
          <path d="M22 40 Q40 27 58 40 Q40 53 22 40 Z" fill="none" stroke="#0a4e2a" stroke-width="2"/>
          <circle cx="40" cy="40" r="6" fill="none" stroke="#0a4e2a" stroke-width="1.5"/>
          <circle cx="40" cy="40" r="2.5" fill="#0a4e2a"/>
        </svg>
      </div>
      <p>> awaiting input... upload image and select folder to begin _</p>
    </div>
  </div>
  </div><!-- /panelImage -->

  <div class="mode-panel" id="panelText">
    <div id="textDisabledNotice" hidden style="text-align:center;padding:40px;color:#ff6e41;">
      <p style="font-size:16px;">> Text scanning unavailable</p>
      <p style="font-size:12px;color:#0a5e2a;margin-top:10px;">Install: pip install faster-whisper</p>
    </div>
    <div id="textControls">
      <div id="audioFilePanel" class="controls">
        <div class="upload-zone" id="audioDropZone" onclick="if(!audioUploadedFile) document.getElementById('audioFileInput').click()">
          <div class="upload-icon">
            <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
              <rect x="12" y="35" width="6" height="30" rx="3" fill="#00ff41" opacity="0.5"/>
              <rect x="24" y="20" width="6" height="60" rx="3" fill="#00ff41" opacity="0.5"/>
              <rect x="36" y="30" width="6" height="40" rx="3" fill="#00ff41" opacity="0.5"/>
              <rect x="48" y="15" width="6" height="70" rx="3" fill="#00ff41" opacity="0.5"/>
              <rect x="60" y="25" width="6" height="50" rx="3" fill="#00ff41" opacity="0.5"/>
              <rect x="72" y="32" width="6" height="36" rx="3" fill="#00ff41" opacity="0.5"/>
              <rect x="84" y="38" width="6" height="24" rx="3" fill="#00ff41" opacity="0.5"/>
            </svg>
          </div>
          <p><strong>[ Click or drag & drop ]</strong></p>
          <p>Upload audio file (mp3, wav, m4a, flac, ogg)</p>
          <input type="file" id="audioFileInput" accept=".mp3,.wav,.m4a,.flac,.ogg,.wma,.aac,.opus" hidden>
        </div>
        <div class="settings">
          <label>> Search Keywords</label>
          <input type="text" class="search-text-input" id="audioSearchText"
                 placeholder="Type words to search for in the audio...">
          <p style="color:#0a5e2a;font-size:11px;margin-bottom:15px;">
            Enter words or phrases to find in the audio transcript
          </p>
          <button class="scan-btn" id="audioScanBtn" onclick="startAudioScan()" disabled>
            [ Transcribe & Search ]
          </button>
        </div>
      </div>
      <div class="history-section">
        <button class="history-toggle" id="textHistoryToggle" onclick="toggleScanHistory('text')">
          > Scan History [ Click to expand ]
        </button>
        <div id="textHistoryPanel" hidden>
          <div class="history-controls" id="textHistoryControls" hidden>
            <button class="history-clear-btn" onclick="clearScanHistory('text')">[ Clear History ]</button>
            <span style="color:#0a5e2a;font-size:11px;" id="textHistoryCount"></span>
          </div>
          <div id="textHistoryList"></div>
        </div>
      </div>
      <div class="status" id="textStatus" hidden>
        <div style="display:flex;align-items:center;justify-content:center;gap:10px;">
          <span class="spinner"></span> <span id="textStatusText">Processing...</span>
        </div>
        <div id="textTimerDisplay" style="text-align:center;margin-top:6px;font-size:20px;color:#00ff41;text-shadow:0 0 10px #00ff41;letter-spacing:2px;">0:00.0</div>
        <div class="progress-bar"><div class="fill" id="textProgressFill" style="width:0%"></div></div>
      </div>
      <div id="textResults">
        <div class="empty-state">
          <div class="icon">
            <svg width="80" height="80" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
              <rect x="12" y="35" width="6" height="30" rx="3" fill="#0a4e2a"/>
              <rect x="24" y="20" width="6" height="60" rx="3" fill="#0a4e2a"/>
              <rect x="36" y="30" width="6" height="40" rx="3" fill="#0a4e2a"/>
              <rect x="48" y="15" width="6" height="70" rx="3" fill="#0a4e2a"/>
              <rect x="60" y="25" width="6" height="50" rx="3" fill="#0a4e2a"/>
              <rect x="72" y="32" width="6" height="36" rx="3" fill="#0a4e2a"/>
              <rect x="84" y="38" width="6" height="24" rx="3" fill="#0a4e2a"/>
            </svg>
          </div>
          <p>> awaiting input... upload audio and enter search terms to begin _</p>
        </div>
      </div>
    </div>
  </div><!-- /panelText -->

  <div class="mode-panel" id="panelAudio">
    <div id="audioDisabledNotice" hidden style="text-align:center;padding:40px;color:#ff6e41;">
      <p style="font-size:16px;">> Audio scanning unavailable</p>
      <p style="font-size:12px;color:#0a5e2a;margin-top:10px;">Install: pip install faster-whisper</p>
    </div>
    <div id="audioControls">
      <div class="controls">
        <div class="settings" style="flex:1;">
          <label>> Scan Folder</label>
          <div class="folder-row">
            <input type="text" id="audioFolderPath" placeholder="Type path or click Browse...">
            <button class="browse-btn" onclick="openBrowseModal('audioFolderPath')">Browse</button>
          </div>
          <p style="color:#0a5e2a;font-size:11px;margin:8px 0 0;">
            Scans all audio files (mp3, wav, m4a, flac, ogg) + audio from video files (mp4, mov, mkv, etc.)
          </p>
        </div>
        <div class="settings" style="flex:1;">
          <label>> Search Keywords</label>
          <input type="text" class="search-text-input" id="audioFolderSearchText"
                 placeholder="Type words to search for...">
          <button class="scan-btn" id="audioFolderScanBtn" onclick="startAudioFolderScan()" disabled>
            [ Scan All Audio ]
          </button>
        </div>
      </div>
      <div class="history-section">
        <button class="history-toggle" id="audioHistoryToggle" onclick="toggleScanHistory('audio')">
          > Scan History [ Click to expand ]
        </button>
        <div id="audioHistoryPanel" hidden>
          <div class="history-controls" id="audioHistoryControls" hidden>
            <button class="history-clear-btn" onclick="clearScanHistory('audio')">[ Clear History ]</button>
            <span style="color:#0a5e2a;font-size:11px;" id="audioHistoryCount"></span>
          </div>
          <div id="audioHistoryList"></div>
        </div>
      </div>
      <div class="status" id="audioStatus" hidden>
        <div style="display:flex;align-items:center;justify-content:center;gap:10px;">
          <span class="spinner"></span> <span id="audioStatusText">Processing...</span>
        </div>
        <div id="audioTimerDisplay" style="text-align:center;margin-top:6px;font-size:20px;color:#00ff41;text-shadow:0 0 10px #00ff41;letter-spacing:2px;">0:00.0</div>
        <div class="progress-bar"><div class="fill" id="audioProgressFill" style="width:0%"></div></div>
      </div>
      <div id="audioResults">
        <div class="empty-state">
          <div class="icon">
            <svg width="80" height="80" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
              <rect x="12" y="35" width="6" height="30" rx="3" fill="#0a4e2a"/>
              <rect x="24" y="20" width="6" height="60" rx="3" fill="#0a4e2a"/>
              <rect x="36" y="30" width="6" height="40" rx="3" fill="#0a4e2a"/>
              <rect x="48" y="15" width="6" height="70" rx="3" fill="#0a4e2a"/>
              <rect x="60" y="25" width="6" height="50" rx="3" fill="#0a4e2a"/>
              <rect x="72" y="32" width="6" height="36" rx="3" fill="#0a4e2a"/>
              <rect x="84" y="38" width="6" height="24" rx="3" fill="#0a4e2a"/>
            </svg>
          </div>
          <p>> awaiting input... select folder and enter search terms to begin _</p>
        </div>
      </div>
    </div>
  </div><!-- /panelAudio -->

  <div class="mode-panel" id="panelTranscribe">
    <div id="transcribeDisabledNotice" hidden style="text-align:center;padding:40px;color:#ff6e41;">
      <p style="font-size:16px;">> Transcription unavailable</p>
      <p style="font-size:12px;color:#0a5e2a;margin-top:10px;">Install: pip install faster-whisper</p>
    </div>
    <div id="transcribeControls">
      <div class="controls">
        <div class="upload-zone" id="transcribeDropZone" onclick="if(!transcribeUploadedFile) document.getElementById('transcribeFileInput').click()">
          <div class="upload-icon">
            <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
              <rect x="12" y="35" width="6" height="30" rx="3" fill="#00ff41" opacity="0.5"/>
              <rect x="24" y="20" width="6" height="60" rx="3" fill="#00ff41" opacity="0.5"/>
              <rect x="36" y="30" width="6" height="40" rx="3" fill="#00ff41" opacity="0.5"/>
              <rect x="48" y="15" width="6" height="70" rx="3" fill="#00ff41" opacity="0.5"/>
              <rect x="60" y="25" width="6" height="50" rx="3" fill="#00ff41" opacity="0.5"/>
              <rect x="72" y="32" width="6" height="36" rx="3" fill="#00ff41" opacity="0.5"/>
              <rect x="84" y="38" width="6" height="24" rx="3" fill="#00ff41" opacity="0.5"/>
            </svg>
          </div>
          <p><strong>[ Click or drag & drop ]</strong></p>
          <p>Upload audio or video file to transcribe</p>
          <input type="file" id="transcribeFileInput" accept=".mp3,.wav,.m4a,.flac,.ogg,.wma,.aac,.opus,.mp4,.mov,.avi,.mkv,.webm,.m4v" hidden>
        </div>
        <div class="settings">
          <label>> Search Keywords (optional)</label>
          <input type="text" class="search-text-input" id="transcribeSearchText"
                 placeholder="Leave empty for full transcript, or type keywords to highlight...">
          <p style="color:#0a5e2a;font-size:11px;margin-bottom:8px;">
            Transcribes the full audio and highlights matching keywords
          </p>
          <label style="display:flex;align-items:center;gap:8px;cursor:pointer;color:#00ff41;font-size:13px;margin-bottom:15px;user-select:none;">
            <input type="checkbox" id="transcribeTimestamps" checked
                   style="accent-color:#00ff41;width:16px;height:16px;cursor:pointer;">
            Show timestamps
          </label>
          <button class="scan-btn" id="transcribeBtn" onclick="startTranscribe()" disabled>
            [ Transcribe ]
          </button>
        </div>
      </div>
      <div class="history-section">
        <button class="history-toggle" id="transcribeHistoryToggle" onclick="toggleScanHistory('transcribe')">
          > Scan History [ Click to expand ]
        </button>
        <div id="transcribeHistoryPanel" hidden>
          <div class="history-controls" id="transcribeHistoryControls" hidden>
            <button class="history-clear-btn" onclick="clearScanHistory('transcribe')">[ Clear History ]</button>
            <span style="color:#0a5e2a;font-size:11px;" id="transcribeHistoryCount"></span>
          </div>
          <div id="transcribeHistoryList"></div>
        </div>
      </div>
      <div class="status" id="transcribeStatus" hidden>
        <div style="display:flex;align-items:center;justify-content:center;gap:10px;">
          <span class="spinner"></span> <span id="transcribeStatusText">Processing...</span>
        </div>
        <div id="transcribeTimerDisplay" style="text-align:center;margin-top:6px;font-size:20px;color:#00ff41;text-shadow:0 0 10px #00ff41;letter-spacing:2px;">0:00.0</div>
        <div class="progress-bar"><div class="fill" id="transcribeProgressFill" style="width:0%"></div></div>
      </div>
      <div id="transcribeResults">
        <div class="empty-state">
          <div class="icon">
            <svg width="80" height="80" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
              <rect x="12" y="35" width="6" height="30" rx="3" fill="#0a4e2a"/>
              <rect x="24" y="20" width="6" height="60" rx="3" fill="#0a4e2a"/>
              <rect x="36" y="30" width="6" height="40" rx="3" fill="#0a4e2a"/>
              <rect x="48" y="15" width="6" height="70" rx="3" fill="#0a4e2a"/>
              <rect x="60" y="25" width="6" height="50" rx="3" fill="#0a4e2a"/>
              <rect x="72" y="32" width="6" height="36" rx="3" fill="#0a4e2a"/>
              <rect x="84" y="38" width="6" height="24" rx="3" fill="#0a4e2a"/>
            </svg>
          </div>
          <p>> awaiting input... upload audio to transcribe _</p>
        </div>
      </div>
    </div>
  </div><!-- /panelTranscribe -->

</div>
</div>

<!-- Folder Browser Modal -->
<div class="modal-overlay" id="folderModal">
  <div class="folder-modal">
    <div class="folder-modal-header">
      <h2>> Select Folder</h2>
      <button class="modal-close" onclick="closeBrowseModal()">[ X ]</button>
    </div>
    <div class="folder-modal-path" id="modalPath">Loading...</div>
    <div class="folder-modal-list" id="modalList"></div>
    <div class="folder-modal-footer">
      <button class="modal-close" onclick="closeBrowseModal()">Cancel</button>
      <button class="modal-select-btn" onclick="selectCurrentFolder()">[ Select This Folder ]</button>
    </div>
  </div>
</div>

<!-- Matrix Rain Script -->
<script>
const canvas = document.getElementById('matrixCanvas');
const ctx = canvas.getContext('2d');
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;
window.addEventListener('resize', () => { canvas.width = window.innerWidth; canvas.height = window.innerHeight; });
const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789@#$%^&*(){}[]|;:<>,.?/~`';
const fontSize = 14;
const columns = Math.floor(canvas.width / fontSize);
const drops = Array(columns).fill(1);
function drawMatrix() {
  ctx.fillStyle = 'rgba(0, 10, 0, 0.05)';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#00ff41';
  ctx.font = fontSize + 'px monospace';
  for (let i = 0; i < drops.length; i++) {
    const text = chars[Math.floor(Math.random() * chars.length)];
    ctx.fillText(text, i * fontSize, drops[i] * fontSize);
    if (drops[i] * fontSize > canvas.height && Math.random() > 0.975) drops[i] = 0;
    drops[i]++;
  }
}
setInterval(drawMatrix, 50);
</script>

<script>
let uploadedFile = null;

const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const preview = document.getElementById('preview');

['dragenter', 'dragover'].forEach(e => dropZone.addEventListener(e, ev => {
  ev.preventDefault(); dropZone.classList.add('dragover');
}));
['dragleave', 'drop'].forEach(e => dropZone.addEventListener(e, ev => {
  ev.preventDefault(); dropZone.classList.remove('dragover');
}));
dropZone.addEventListener('drop', ev => { handleFile(ev.dataTransfer.files[0]); });
fileInput.addEventListener('change', ev => { if (ev.target.files[0]) handleFile(ev.target.files[0]); });

function handleFile(file) {
  if (!file || !file.type.startsWith('image/')) return;
  uploadedFile = file;
  const reader = new FileReader();
  reader.onload = e => {
    preview.src = e.target.result;
    preview.hidden = false;
    const ico = dropZone.querySelector('.upload-icon');
    if (ico) ico.hidden = true;
    dropZone.querySelectorAll('p').forEach(p => p.hidden = true);
    let info = document.getElementById('fileInfo');
    if (!info) {
      info = document.createElement('p');
      info.id = 'fileInfo';
      info.style.cssText = 'color:#00ff41;margin-top:8px;font-size:12px;';
      dropZone.appendChild(info);
    }
    info.hidden = false;
    info.innerHTML = '> IMAGE LOCKED: <strong>' + escapeHtml(file.name) + '</strong> (' + (file.size/1024).toFixed(0) + ' KB)';
    let changeBtn = document.getElementById('changeBtn');
    if (!changeBtn) {
      changeBtn = document.createElement('button');
      changeBtn.id = 'changeBtn';
      changeBtn.className = 'change-btn';
      changeBtn.textContent = '[ Change Image ]';
      changeBtn.onclick = function(e) {
        e.stopPropagation();
        document.getElementById('fileInput').click();
      };
      dropZone.appendChild(changeBtn);
    }
    changeBtn.hidden = false;
    dropZone.classList.add('has-image');
    updateScanBtn();
  };
  reader.readAsDataURL(file);
}

// Folder path
const folderPath = document.getElementById('folderPath');
folderPath.addEventListener('input', updateScanBtn);

// --- Folder Browser Modal ---
let currentBrowsePath = '';
let browseTargetInput = 'folderPath';

async function openBrowseModal(targetInputId) {
  browseTargetInput = targetInputId || 'folderPath';
  const modal = document.getElementById('folderModal');
  modal.classList.add('open');
  await navigateFolder('~');
}

function closeBrowseModal() {
  document.getElementById('folderModal').classList.remove('open');
}

function selectCurrentFolder() {
  if (currentBrowsePath) {
    const el = document.getElementById(browseTargetInput);
    if (el) { el.value = currentBrowsePath; el.dispatchEvent(new Event('input')); }
    updateScanBtn();
    updateAudioFolderScanBtn();
  }
  closeBrowseModal();
}

async function navigateFolder(path) {
  const list = document.getElementById('modalList');
  const pathEl = document.getElementById('modalPath');
  list.innerHTML = '<div style="padding:20px;text-align:center;color:#0a6e2a;">Loading...</div>';
  try {
    const resp = await fetch('/browse/list', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({path: path})
    });
    const data = await resp.json();
    if (data.error) {
      list.innerHTML = '<div style="padding:20px;text-align:center;color:#ff6e41;">' + escapeHtml(data.error) + '</div>';
      return;
    }
    currentBrowsePath = data.current;
    pathEl.textContent = '> ' + data.current;
    let html = '';
    if (data.parent) {
      html += '<div class="folder-item parent" onclick="navigateFolder(\'' + escapeJs(data.parent) + '\')">' +
        '<span class="icon">..</span> <span>[ Parent Directory ]</span></div>';
    }
    for (const dir of data.dirs) {
      html += '<div class="folder-item" onclick="navigateFolder(\'' + escapeJs(dir.path) + '\')">' +
        '<span class="icon">&#128193;</span> <span>' + escapeHtml(dir.name) + '</span></div>';
    }
    if (data.dirs.length === 0 && !data.parent) {
      html += '<div style="padding:20px;text-align:center;color:#0a4e2a;">No subdirectories</div>';
    }
    if (data.dirs.length === 0 && data.parent) {
      html += '<div style="padding:20px;text-align:center;color:#0a4e2a;">No subdirectories — select this folder or go back</div>';
    }
    list.innerHTML = html;
  } catch (err) {
    list.innerHTML = '<div style="padding:20px;text-align:center;color:#ff6e41;">Error: ' + escapeHtml(err.message) + '</div>';
  }
}

function updateScanBtn() {
  document.getElementById('scanBtn').disabled = !(uploadedFile && folderPath.value.trim());
}

// Timer
let scanTimer = null;
let scanStartTime = 0;
function startTimer() {
  scanStartTime = performance.now();
  const timerEl = document.getElementById('timerDisplay');
  function tick() {
    const elapsed = (performance.now() - scanStartTime) / 1000;
    const mins = Math.floor(elapsed / 60);
    const secs = (elapsed % 60).toFixed(1);
    timerEl.textContent = mins + ':' + (secs < 10 ? '0' : '') + secs;
    scanTimer = requestAnimationFrame(tick);
  }
  tick();
}
function stopTimer() {
  if (scanTimer) { cancelAnimationFrame(scanTimer); scanTimer = null; }
}

// Scan with live progress polling
async function startScan() {
  const btn = document.getElementById('scanBtn');
  const status = document.getElementById('status');
  const results = document.getElementById('results');
  btn.disabled = true;
  status.hidden = false;
  document.getElementById('statusText').textContent = 'Uploading image...';
  document.getElementById('progressFill').style.width = '0%';
  document.getElementById('timerDisplay').textContent = '0:00.0';
  results.innerHTML = '';
  startTimer();
  const formData = new FormData();
  formData.append('image', uploadedFile);
  formData.append('folder', folderPath.value.trim());
  formData.append('threshold', '10');
  try {
    // Step 1: Start scan (returns scan_id immediately)
    const startResp = await fetch('/scan', { method: 'POST', body: formData });
    const startData = await startResp.json();
    if (startData.error) {
      stopTimer();
      status.hidden = true;
      results.innerHTML = '<div class="empty-state"><div class="icon">&#9888;</div><p>' +
        escapeHtml(startData.error) + '</p></div>';
      btn.disabled = false;
      return;
    }
    const scanId = startData.scan_id;
    document.getElementById('statusText').textContent = 'Scanning... 0% | 0 matches found';

    // Live results container — show matches as they stream in
    results.innerHTML = '<div id="liveResults" class="results-grid"></div>';

    // Step 2: Poll progress until done, rendering matches in real-time
    const data = await new Promise((resolve, reject) => {
      const poll = async () => {
        try {
          const resp = await fetch('/scan/progress?id=' + scanId);
          const prog = await resp.json();
          if (prog.error && !prog.done) {
            reject(new Error(prog.error));
            return;
          }
          // Render any new matches that arrived
          if (prog.new_matches && prog.new_matches.length > 0) {
            const live = document.getElementById('liveResults');
            prog.new_matches.forEach(m => {
              const card = document.createElement('div');
              card.className = 'result-card';
              card.style.animation = 'fadeIn 0.3s ease-in';
              const label = m.type === 'video' ? '&#127909; VIDEO @ ' + m.timestamp + 's' : '&#128444; IMAGE';
              card.innerHTML = '<img src="data:image/jpeg;base64,' + m.thumbnail + '" class="result-thumb" onclick="window.open(this.src)">' +
                '<div class="result-info">' +
                '<div class="result-path" title="' + escapeHtml(m.path) + '">' + escapeHtml(m.path.split('/').pop()) + '</div>' +
                '<div style="color:#7fff00;font-size:11px;">' + label + ' | Distance: ' + m.distance + '</div>' +
                '<div style="color:#0a6e2a;font-size:10px;word-break:break-all;">' + escapeHtml(m.folder) + '</div>' +
                '</div>';
              live.appendChild(card);
            });
          }
          if (prog.done) {
            if (prog.error) { reject(new Error(prog.error)); return; }
            document.getElementById('progressFill').style.width = '100%';
            document.getElementById('statusText').textContent = 'Complete! Processing results...';
            resolve(prog);
            return;
          }
          // Update progress bar and status text
          const pct = prog.total > 0 ? Math.round((prog.processed / prog.total) * 100) : 0;
          document.getElementById('progressFill').style.width = pct + '%';
          let statusMsg = 'Scanning... ' + pct + '%';
          statusMsg += ' (' + prog.processed + '/' + prog.total + ')';
          statusMsg += ' | ' + prog.matches + ' match' + (prog.matches !== 1 ? 'es' : '') + ' found';
          document.getElementById('statusText').textContent = statusMsg;
          setTimeout(poll, 300);
        } catch (err) { reject(err); }
      };
      poll();
    });

    stopTimer();
    status.hidden = true;
    if (data.error) {
      results.innerHTML = '<div class="empty-state"><div class="icon">&#9888;</div><p>' +
        escapeHtml(data.error) + '</p></div>';
      btn.disabled = false;
      return;
    }
    const matches = data.results;
    if (matches.length === 0) {
      results.innerHTML = '<div class="empty-state"><div class="icon">&#128533;</div>' +
        '<p>No matching images or video frames found in this folder.</p></div>';
    } else {
      const imgCount = data.total_images_scanned || 0;
      const vidCount = data.total_videos_scanned || 0;
      let scannedText = imgCount + ' images';
      if (vidCount > 0) scannedText += ' + ' + vidCount + ' videos';
      let html = '<div class="results-header">Found ' + matches.length + ' match(es) in ' +
        data.elapsed.toFixed(1) + 's (' + scannedText + ' scanned' +
        (data.errors > 0 ? ', ' + data.errors + ' skipped' : '') + ')</div><div class="results">';
      for (const m of matches) {
        const pct = Math.max(0, (1 - m.distance / 64) * 100).toFixed(1);
        let cls = m.distance <= 5 ? 'high' : m.distance <= 10 ? 'med' : 'low';
        let badge = '';
        let timestampHtml = '';
        if (m.type === 'video') {
          badge = '<span class="video-badge">VIDEO</span>';
          const totalSec = m.timestamp || 0;
          const mins = Math.floor(totalSec / 60);
          const secs = (totalSec % 60).toFixed(1);
          timestampHtml = '<div class="timestamp">Frame match at ' + mins + ':' + (secs < 10 ? '0' : '') + secs + '</div>';
        }
        let simText = badge + (m.distance === 0 ? 'Identical match' : pct + '% similar (distance: ' + m.distance + ')');
        const thumbSrc = m.thumbnail ? 'data:image/jpeg;base64,' + m.thumbnail : '';
        const thumbHtml = thumbSrc ? '<img src="' + thumbSrc + '">' : '<div style="width:120px;height:120px;background:#333;border-radius:4px;display:flex;align-items:center;justify-content:center;color:#666">Error</div>';
        html += '<div class="result-card">' + thumbHtml +
          '<div class="result-info"><div class="similarity ' + cls + '">' + simText + '</div>' +
          '<div class="filepath">' + escapeHtml(m.path) + '</div>' +
          timestampHtml +
          '<div class="folder">Folder: ' + escapeHtml(m.folder) + '</div></div>' +
          '<button class="reveal-btn" onclick="reveal(\'' + escapeJs(m.path) + '\')">Reveal in Finder</button></div>';
      }
      html += '</div>';
      results.innerHTML = html;
    }
  } catch (err) {
    stopTimer();
    status.hidden = true;
    results.innerHTML = '<div class="empty-state"><div class="icon">&#9888;</div><p>Error: ' +
      escapeHtml(err.message) + '</p></div>';
  }
  btn.disabled = false;
}

async function reveal(path) {
  await fetch('/reveal', { method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({path: path}) });
}

function escapeHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function escapeJs(s) {
  return s.replace(/\\/g,'\\\\').replace(/'/g,"\\'");
}

// --- History ---
let historyOpen = false;
async function toggleHistory() {
  const panel = document.getElementById('historyPanel');
  const btn = document.getElementById('historyToggle');
  historyOpen = !historyOpen;
  if (historyOpen) {
    panel.hidden = false;
    btn.innerHTML = '> Upload History [ Click to collapse ]';
    await loadHistory();
  } else {
    panel.hidden = true;
    btn.innerHTML = '> Upload History [ Click to expand ]';
  }
}

async function loadHistory() {
  const grid = document.getElementById('historyGrid');
  const pathEl = document.getElementById('historyPath');
  const controls = document.getElementById('historyControls');
  const countEl = document.getElementById('historyCount');
  try {
    const resp = await fetch('/history');
    const data = await resp.json();
    pathEl.textContent = '> Saved to: ' + data.folder;
    if (data.items.length === 0) {
      grid.innerHTML = '<p style="color:#0a4e2a;grid-column:1/-1;text-align:center;">No history yet. Upload an image and scan to start.</p>';
      controls.hidden = true;
      return;
    }
    controls.hidden = false;
    countEl.textContent = data.items.length + ' item' + (data.items.length !== 1 ? 's' : '');
    let html = '';
    for (const item of data.items) {
      const thumbSrc = item.thumbnail ? 'data:image/jpeg;base64,' + item.thumbnail : '';
      const imgTag = thumbSrc ? '<img src="' + thumbSrc + '">' : '<div style="height:80px;background:#0a2e1a;display:flex;align-items:center;justify-content:center;color:#0a4e2a;font-size:10px">No preview</div>';
      const safeP = escapeJs(item.path);
      const safeF = escapeJs(item.filename);
      html += '<div class="history-item" onclick="useHistoryImage(\'' + safeP + '\',\'' + safeF + '\',' + item.size + ')" title="Click to use as reference">' +
        '<span class="hist-delete" onclick="event.stopPropagation();deleteHistoryItem(\'' + safeP + '\',this)" title="Delete">&times;</span>' +
        imgTag + '<div class="hist-name">' + escapeHtml(item.filename) + '</div></div>';
    }
    grid.innerHTML = html;
  } catch (err) {
    grid.innerHTML = '<p style="color:#ff6e41;grid-column:1/-1;text-align:center;">Error loading history</p>';
  }
}

async function deleteHistoryItem(path, btn) {
  const item = btn.closest('.history-item');
  item.style.opacity = '0.3';
  try {
    const resp = await fetch('/history/delete', { method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({path: path}) });
    const data = await resp.json();
    if (data.ok) {
      item.remove();
      // Update count
      const grid = document.getElementById('historyGrid');
      const remaining = grid.querySelectorAll('.history-item').length;
      const countEl = document.getElementById('historyCount');
      if (remaining === 0) {
        grid.innerHTML = '<p style="color:#0a4e2a;grid-column:1/-1;text-align:center;">No history yet. Upload an image and scan to start.</p>';
        document.getElementById('historyControls').hidden = true;
      } else {
        countEl.textContent = remaining + ' item' + (remaining !== 1 ? 's' : '');
      }
    } else {
      item.style.opacity = '1';
    }
  } catch (err) {
    item.style.opacity = '1';
  }
}

async function clearAllHistory() {
  if (!confirm('Delete all history images? This cannot be undone.')) return;
  try {
    const resp = await fetch('/history/clear', { method: 'POST' });
    const data = await resp.json();
    if (data.ok) {
      await loadHistory();
    }
  } catch (err) {
    console.error('Failed to clear history', err);
  }
}

async function useHistoryImage(path, filename, size) {
  try {
    const resp = await fetch('/history/use', { method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({path: path}) });
    if (!resp.ok) return;
    const blob = await resp.blob();
    uploadedFile = new File([blob], filename, {type: blob.type});
    const url = URL.createObjectURL(blob);
    preview.src = url;
    preview.hidden = false;
    const ico = dropZone.querySelector('.upload-icon');
    if (ico) ico.hidden = true;
    dropZone.querySelectorAll('p').forEach(p => p.hidden = true);
    let info = document.getElementById('fileInfo');
    if (!info) {
      info = document.createElement('p');
      info.id = 'fileInfo';
      info.style.cssText = 'color:#00ff41;margin-top:8px;font-size:12px;';
      dropZone.appendChild(info);
    }
    info.hidden = false;
    info.innerHTML = '> IMAGE LOCKED: <strong>' + escapeHtml(filename) + '</strong> (' + (size/1024).toFixed(0) + ' KB) [from history]';
    dropZone.classList.add('has-image');
    let changeBtn = document.getElementById('changeBtn');
    if (!changeBtn) {
      changeBtn = document.createElement('button');
      changeBtn.id = 'changeBtn';
      changeBtn.className = 'change-btn';
      changeBtn.textContent = '[ Change Image ]';
      changeBtn.onclick = function(e) {
        e.stopPropagation();
        document.getElementById('fileInput').click();
      };
      dropZone.appendChild(changeBtn);
    }
    changeBtn.hidden = false;
    updateScanBtn();
  } catch (err) {
    console.error('Failed to load history image', err);
  }
}

// ==================== MODE SWITCHING ====================
function switchMode(mode) {
  document.querySelectorAll('.top-mode-tabs .mode-tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.mode-panel').forEach(p => p.classList.remove('active'));
  document.getElementById('tab' + mode.charAt(0).toUpperCase() + mode.slice(1)).classList.add('active');
  document.getElementById('panel' + mode.charAt(0).toUpperCase() + mode.slice(1)).classList.add('active');
}

// Check audio availability on load
(async function checkAudioStatus() {
  try {
    const resp = await fetch('/audio/status');
    const data = await resp.json();
    if (!data.available) {
      document.getElementById('textDisabledNotice').hidden = false;
      document.getElementById('textControls').style.display = 'none';
      document.getElementById('audioDisabledNotice').hidden = false;
      document.getElementById('audioControls').style.display = 'none';
      document.getElementById('transcribeDisabledNotice').hidden = false;
      document.getElementById('transcribeControls').style.display = 'none';
    }
  } catch (e) {}
})();

// ==================== AUDIO FILE UPLOAD ====================
let audioUploadedFile = null;
const audioDropZone = document.getElementById('audioDropZone');
const audioFileInput = document.getElementById('audioFileInput');

if (audioDropZone) {
  ['dragenter','dragover'].forEach(e => audioDropZone.addEventListener(e, ev => {
    ev.preventDefault(); audioDropZone.classList.add('dragover');
  }));
  ['dragleave','drop'].forEach(e => audioDropZone.addEventListener(e, ev => {
    ev.preventDefault(); audioDropZone.classList.remove('dragover');
  }));
  audioDropZone.addEventListener('drop', ev => { handleAudioFile(ev.dataTransfer.files[0]); });
  audioFileInput.addEventListener('change', ev => { if (ev.target.files[0]) handleAudioFile(ev.target.files[0]); });
}

function handleAudioFile(file) {
  if (!file) return;
  const validExts = ['.mp3','.wav','.m4a','.flac','.ogg','.wma','.aac','.opus'];
  const ext = '.' + file.name.split('.').pop().toLowerCase();
  if (!validExts.includes(ext)) { alert('Unsupported format. Use: ' + validExts.join(', ')); return; }
  audioUploadedFile = file;
  const ico = audioDropZone.querySelector('.upload-icon');
  if (ico) ico.hidden = true;
  audioDropZone.querySelectorAll(':scope > p').forEach(p => p.hidden = true);
  let info = document.getElementById('audioFileInfo');
  if (!info) {
    info = document.createElement('p');
    info.id = 'audioFileInfo';
    info.style.cssText = 'color:#00ff41;margin-top:8px;font-size:12px;';
    audioDropZone.appendChild(info);
  }
  info.hidden = false;
  const sizeMB = (file.size / (1024*1024)).toFixed(1);
  info.innerHTML = '> AUDIO LOADED: <strong>' + escapeHtml(file.name) + '</strong> (' + sizeMB + ' MB)';
  let changeBtn = document.getElementById('audioChangeBtn');
  if (!changeBtn) {
    changeBtn = document.createElement('button');
    changeBtn.id = 'audioChangeBtn';
    changeBtn.className = 'change-btn';
    changeBtn.textContent = '[ Change Audio ]';
    changeBtn.onclick = function(e) { e.stopPropagation(); audioFileInput.click(); };
    audioDropZone.appendChild(changeBtn);
  }
  changeBtn.hidden = false;
  audioDropZone.classList.add('has-image');
  updateAudioScanBtn();
}

const audioSearchText = document.getElementById('audioSearchText');
if (audioSearchText) audioSearchText.addEventListener('input', updateAudioScanBtn);

function updateAudioScanBtn() {
  const btn = document.getElementById('audioScanBtn');
  if (btn) btn.disabled = !(audioUploadedFile && document.getElementById('audioSearchText').value.trim());
}

// ==================== (Audio sub-tabs removed — now separate top-level panels) ====================

// ==================== AUDIO FOLDER SCAN ====================
const audioFolderPath = document.getElementById('audioFolderPath');
const audioFolderSearchText = document.getElementById('audioFolderSearchText');
if (audioFolderPath) audioFolderPath.addEventListener('input', updateAudioFolderScanBtn);
if (audioFolderSearchText) audioFolderSearchText.addEventListener('input', updateAudioFolderScanBtn);

function updateAudioFolderScanBtn() {
  const btn = document.getElementById('audioFolderScanBtn');
  if (btn) btn.disabled = !(
    document.getElementById('audioFolderPath').value.trim() &&
    document.getElementById('audioFolderSearchText').value.trim()
  );
}

async function startAudioFolderScan() {
  const btn = document.getElementById('audioFolderScanBtn');
  const status = document.getElementById('audioStatus');
  const results = document.getElementById('audioResults');
  const folder = document.getElementById('audioFolderPath').value.trim();
  const searchText = document.getElementById('audioFolderSearchText').value.trim();
  btn.disabled = true;
  status.hidden = false;
  document.getElementById('audioStatusText').textContent = 'Scanning folder...';
  document.getElementById('audioProgressFill').style.width = '0%';
  document.getElementById('audioTimerDisplay').textContent = '0:00.0';
  results.innerHTML = '';
  startAudioTimer();

  try {
    const startResp = await fetch('/audio/folder-scan', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({folder: folder, search_text: searchText})
    });
    const startData = await startResp.json();
    if (startData.error) {
      stopAudioTimer(); status.hidden = true;
      results.innerHTML = '<div class="empty-state"><p style="color:#ff6e41;">' + escapeHtml(startData.error) + '</p></div>';
      btn.disabled = false; return;
    }
    const scanId = startData.scan_id;

    // Live results container for streaming file results
    results.innerHTML = '<div id="audioLiveFiles"></div>';

    const data = await new Promise((resolve, reject) => {
      const poll = async () => {
        try {
          const resp = await fetch('/audio/scan/progress?id=' + scanId);
          const prog = await resp.json();
          // Render new partial results live
          if (prog.new_matches && prog.new_matches.length > 0) {
            const liveBox = document.getElementById('audioLiveFiles');
            prog.new_matches.forEach(m => {
              if (m.kind === 'file_done' && liveBox) {
                const div = document.createElement('div');
                div.style.cssText = 'margin-bottom:12px;border:1px solid #0a3e1a;border-radius:6px;padding:10px;background:rgba(0,10,0,0.3);animation:fadeIn 0.3s ease-in;';
                const badge = m.is_video ? '<span class="audio-badge" style="background:#e54cff;">VIDEO</span>' : '<span class="audio-badge">AUDIO</span>';
                const dur = m.audio_duration ? fmtTime(m.audio_duration) : '';
                let inner = '<div style="font-size:13px;color:#00ff41;margin-bottom:6px;">' + badge + ' <strong>' + escapeHtml(m.filename) + '</strong>';
                inner += ' <span style="color:#0a5e2a;font-size:11px;">(' + (dur ? dur + ' / ' : '') + m.matches_count + ' match' + (m.matches_count !== 1 ? 'es' : '') + ')</span>';
                inner += ' <button style="float:right;background:none;border:1px solid #0a5e2a;color:#0a8e3a;border-radius:3px;cursor:pointer;font-family:inherit;font-size:11px;padding:2px 8px;" onclick="reveal(\'' + escapeJs(m.file) + '\')">Reveal</button>';
                inner += '</div>';
                if (m.matches && m.matches.length > 0) {
                  m.matches.forEach(mt => {
                    inner += '<div class="transcript-segment" style="margin-bottom:4px;">';
                    inner += '<div class="seg-time">' + fmtTime(mt.start) + ' — ' + fmtTime(mt.end) + '</div>';
                    inner += '<div class="seg-text">' + mt.highlighted_text + '</div>';
                    inner += '<div class="matched-terms">Matched: ' + mt.matched_terms.map(t => escapeHtml(t)).join(', ') + '</div></div>';
                  });
                }
                div.innerHTML = inner;
                liveBox.appendChild(div);
              } else if (m.kind === 'match' && liveBox) {
                // Individual match from current file (already handled by file_done, but show immediately)
              }
            });
          }
          if (prog.done) {
            if (prog.error) { reject(new Error(prog.error)); return; }
            document.getElementById('audioProgressFill').style.width = '100%';
            resolve(prog); return;
          }
          document.getElementById('audioProgressFill').style.width = prog.percent + '%';
          document.getElementById('audioStatusText').textContent = prog.phase_detail || 'Processing...';
          setTimeout(poll, 400);
        } catch (err) { reject(err); }
      };
      poll();
    });

    stopAudioTimer(); status.hidden = true;
    renderAudioFolderResults(data);
  } catch (err) {
    stopAudioTimer(); status.hidden = true;
    results.innerHTML = '<div class="empty-state"><p style="color:#ff6e41;">Error: ' + escapeHtml(err.message) + '</p></div>';
  }
  btn.disabled = false;
  updateAudioFolderScanBtn();
}

function renderAudioFolderResults(data) {
  const results = document.getElementById('audioResults');
  const fileResults = data.file_results || [];
  const elapsed = data.elapsed || '?';
  const totalFiles = data.total_files || 0;
  const totalMatches = data.total_matches || 0;

  let html = '<div class="results-header" style="margin-bottom:15px;padding:10px;border:1px solid #0a3e1a;border-radius:4px;color:#0a8e3a;font-size:13px;">';
  html += '> Scanned <span style="color:#00ff41;">' + totalFiles + '</span> file(s)';
  html += ' &bull; <span style="color:#00ff41;">' + totalMatches + '</span> match(es)';
  html += ' for "<span style="color:#00ff41;">' + escapeHtml(data.search_text) + '</span>"';
  html += ' &bull; ' + elapsed + 's</div>';

  if (totalMatches === 0 && totalFiles > 0) {
    html += '<div class="empty-state"><p>> No matches found in any file. Try different keywords.</p></div>';
  }

  for (const fr of fileResults) {
    if (fr.error) continue;
    if (!fr.matches || fr.matches.length === 0) continue;

    const badge = fr.is_video ? '<span class="audio-badge" style="background:#e54cff;">VIDEO</span>' : '<span class="audio-badge">AUDIO</span>';
    const dur = fr.audio_duration ? fmtTime(fr.audio_duration) : '??';
    const lang = fr.language ? fr.language.toUpperCase() : '';

    html += '<div style="margin-bottom:20px;border:1px solid #0a3e1a;border-radius:6px;padding:12px;background:rgba(0,10,0,0.3);">';
    html += '<div style="margin-bottom:10px;font-size:13px;color:#00ff41;">' + badge + ' <strong>' + escapeHtml(fr.filename) + '</strong>';
    html += ' <span style="color:#0a5e2a;font-size:11px;">(' + dur + (lang ? ' / ' + lang : '') + ' / ' + fr.matches.length + ' match' + (fr.matches.length !== 1 ? 'es' : '') + ')</span>';
    html += ' <button style="float:right;background:none;border:1px solid #0a5e2a;color:#0a8e3a;border-radius:3px;cursor:pointer;font-family:inherit;font-size:11px;padding:2px 8px;" onclick="reveal(\'' + escapeJs(fr.file) + '\')">Reveal</button>';
    html += '</div>';

    for (const m of fr.matches) {
      html += '<div class="transcript-segment">';
      html += '<div class="seg-time">' + fmtTime(m.start) + ' — ' + fmtTime(m.end) + '</div>';
      html += '<div class="seg-text">' + m.highlighted_text + '</div>';
      html += '<div class="matched-terms">Matched: ' + m.matched_terms.map(t => escapeHtml(t)).join(', ') + '</div></div>';
    }

    // Collapsible full transcript per file
    if (fr.full_transcript && fr.full_transcript.length > 0) {
      const tid = 'ft_' + fr.filename.replace(/[^a-zA-Z0-9]/g, '_');
      html += '<button class="history-toggle" style="font-size:11px;margin-top:6px;" onclick="document.getElementById(\'' + tid + '\').hidden=!document.getElementById(\'' + tid + '\').hidden">';
      html += '> Transcript (' + fr.total_segments + ' segments)</button>';
      html += '<div class="full-transcript" id="' + tid + '" hidden style="max-height:200px;">';
      for (const seg of fr.full_transcript) {
        html += '<div style="margin-bottom:4px;"><span style="color:#7fff00;font-size:10px;">[' + fmtTime(seg.start) + ']</span> ' + escapeHtml(seg.text) + '</div>';
      }
      html += '</div>';
    }

    html += '</div>';
  }

  if (totalFiles === 0) {
    html += '<div class="empty-state"><p>> No audio or video files found in the selected folder.</p></div>';
  }

  results.innerHTML = html;
}

// ==================== AUDIO SCAN + POLLING ====================
let audioTimerRAF = null;
let audioScanStart = 0;

function startAudioTimer(panel) {
  audioScanStart = performance.now();
  const elId = panel === 'text' ? 'textTimerDisplay' : panel === 'transcribe' ? 'transcribeTimerDisplay' : 'audioTimerDisplay';
  const el = document.getElementById(elId);
  function tick() {
    const s = (performance.now() - audioScanStart) / 1000;
    const m = Math.floor(s / 60);
    const sec = (s % 60).toFixed(1);
    el.textContent = m + ':' + (sec < 10 ? '0' : '') + sec;
    audioTimerRAF = requestAnimationFrame(tick);
  }
  tick();
}
function stopAudioTimer() { if (audioTimerRAF) { cancelAnimationFrame(audioTimerRAF); audioTimerRAF = null; } }

async function startAudioScan() {
  const btn = document.getElementById('audioScanBtn');
  const status = document.getElementById('textStatus');
  const results = document.getElementById('textResults');
  btn.disabled = true;
  status.hidden = false;
  document.getElementById('textStatusText').textContent = 'Uploading audio...';
  document.getElementById('textProgressFill').style.width = '0%';
  document.getElementById('textTimerDisplay').textContent = '0:00.0';
  results.innerHTML = '';
  startAudioTimer('text');

  const fd = new FormData();
  fd.append('audio', audioUploadedFile);
  fd.append('search_text', document.getElementById('audioSearchText').value.trim());
  fd.append('scan_type', 'text');

  try {
    const startResp = await fetch('/audio/scan', { method: 'POST', body: fd });
    const startData = await startResp.json();
    if (startData.error) {
      stopAudioTimer('text'); status.hidden = true;
      results.innerHTML = '<div class="empty-state"><p style="color:#ff6e41;">' + escapeHtml(startData.error) + '</p></div>';
      btn.disabled = false; return;
    }
    const scanId = startData.scan_id;
    document.getElementById('textStatusText').textContent = 'Processing audio...';

    // Live results container for streaming
    results.innerHTML = '<div id="textLiveMatches"></div><div id="textLiveSegments" style="margin-top:10px;"></div>';

    const data = await new Promise((resolve, reject) => {
      const poll = async () => {
        try {
          const resp = await fetch('/audio/scan/progress?id=' + scanId);
          const prog = await resp.json();
          // Render new partial results live
          if (prog.new_matches && prog.new_matches.length > 0) {
            const matchBox = document.getElementById('textLiveMatches');
            const segBox = document.getElementById('textLiveSegments');
            prog.new_matches.forEach(m => {
              if (m.kind === 'match') {
                const div = document.createElement('div');
                div.className = 'transcript-segment';
                div.style.animation = 'fadeIn 0.3s ease-in';
                div.innerHTML = '<div class="seg-time"><span class="audio-badge">MATCH</span> ' +
                  fmtTime(m.start) + ' — ' + fmtTime(m.end) + '</div>' +
                  '<div class="seg-text">' + m.highlighted_text + '</div>' +
                  '<div class="matched-terms">Matched: ' + m.matched_terms.map(t => escapeHtml(t)).join(', ') + '</div>';
                if (matchBox) matchBox.appendChild(div);
              } else if (m.kind === 'segment') {
                const div = document.createElement('div');
                div.style.cssText = 'margin-bottom:4px;animation:fadeIn 0.2s ease-in;font-size:12px;color:#0a8e3a;';
                div.innerHTML = '<span style="color:#7fff00;font-size:10px;">[' + fmtTime(m.start) + ']</span> ' + escapeHtml(m.text);
                if (segBox) segBox.appendChild(div);
              }
            });
          }
          if (prog.done) {
            if (prog.error) { reject(new Error(prog.error)); return; }
            document.getElementById('textProgressFill').style.width = '100%';
            resolve(prog); return;
          }
          document.getElementById('textProgressFill').style.width = prog.percent + '%';
          document.getElementById('textStatusText').textContent = prog.phase_detail || 'Processing...';
          setTimeout(poll, 400);
        } catch (err) { reject(err); }
      };
      poll();
    });

    stopAudioTimer('text'); status.hidden = true;
    renderAudioResults(data);
  } catch (err) {
    stopAudioTimer('text'); status.hidden = true;
    results.innerHTML = '<div class="empty-state"><p style="color:#ff6e41;">Error: ' + escapeHtml(err.message) + '</p></div>';
  }
  btn.disabled = false;
  updateAudioScanBtn();
}

// ==================== RENDER AUDIO RESULTS ====================
function fmtTime(s) {
  const m = Math.floor(s / 60);
  const sec = (s % 60).toFixed(1);
  return m + ':' + (sec < 10 ? '0' : '') + sec;
}

function renderAudioResults(data) {
  const results = document.getElementById('textResults');
  const matches = data.matches || [];
  const lang = data.language ? data.language.toUpperCase() : '??';
  const dur = data.audio_duration ? fmtTime(data.audio_duration) : '??';
  const elapsed = data.elapsed || '?';

  let html = '<div class="results-header" style="margin-bottom:15px;padding:10px;border:1px solid #0a3e1a;border-radius:4px;color:#0a8e3a;font-size:13px;">';
  if (matches.length === 0) {
    html += '> No matches found for "<span style="color:#00ff41;">' + escapeHtml(data.search_text) + '</span>"';
    html += ' &bull; ' + elapsed + 's &bull; ' + lang + ' &bull; Duration: ' + dur;
    html += '</div>';
    html += '<div class="empty-state"><p>> No segments matched your search terms. Try different keywords or check the full transcript below.</p></div>';
  } else {
    html += '> Found <span style="color:#00ff41;">' + matches.length + '</span> matching segment(s) for "<span style="color:#00ff41;">' + escapeHtml(data.search_text) + '</span>"';
    html += ' &bull; ' + elapsed + 's &bull; ' + lang + ' &bull; Duration: ' + dur;
    html += '</div>';
    for (const m of matches) {
      html += '<div class="transcript-segment">' +
        '<div class="seg-time"><span class="audio-badge">AUDIO</span> ' +
        fmtTime(m.start) + ' — ' + fmtTime(m.end) + '</div>' +
        '<div class="seg-text">' + m.highlighted_text + '</div>' +
        '<div class="matched-terms">Matched: ' + m.matched_terms.map(t => escapeHtml(t)).join(', ') + '</div></div>';
    }
  }

  if (data.full_transcript && data.full_transcript.length > 0) {
    html += '<div style="margin-top:20px;">';
    html += '<button class="history-toggle" onclick="document.getElementById(\'fullTranscriptPanel\').hidden=!document.getElementById(\'fullTranscriptPanel\').hidden">';
    html += '> Full Transcript (' + data.total_segments + ' segments) [ Click to expand ]</button>';
    html += '<div class="full-transcript" id="fullTranscriptPanel" hidden>';
    for (const seg of data.full_transcript) {
      html += '<div style="margin-bottom:8px;"><span style="color:#7fff00;font-size:11px;">[' + fmtTime(seg.start) + ']</span> ' + escapeHtml(seg.text) + '</div>';
    }
    html += '</div></div>';
  }

  results.innerHTML = html;
}

// ==================== TRANSCRIBE TAB ====================
let transcribeUploadedFile = null;
const transcribeDropZone = document.getElementById('transcribeDropZone');
const transcribeFileInput = document.getElementById('transcribeFileInput');

if (transcribeDropZone) {
  ['dragenter','dragover'].forEach(e => transcribeDropZone.addEventListener(e, ev => {
    ev.preventDefault(); transcribeDropZone.classList.add('dragover');
  }));
  ['dragleave','drop'].forEach(e => transcribeDropZone.addEventListener(e, ev => {
    ev.preventDefault(); transcribeDropZone.classList.remove('dragover');
  }));
  transcribeDropZone.addEventListener('drop', ev => { handleTranscribeFile(ev.dataTransfer.files[0]); });
}
if (transcribeFileInput) {
  transcribeFileInput.addEventListener('change', () => { handleTranscribeFile(transcribeFileInput.files[0]); });
}

function handleTranscribeFile(file) {
  if (!file) return;
  const validExts = ['.mp3','.wav','.m4a','.flac','.ogg','.wma','.aac','.opus','.mp4','.mov','.avi','.mkv','.webm','.m4v'];
  const ext = '.' + file.name.split('.').pop().toLowerCase();
  if (!validExts.includes(ext)) { alert('Unsupported format. Use: ' + validExts.join(', ')); return; }
  transcribeUploadedFile = file;
  const ico = transcribeDropZone.querySelector('.upload-icon');
  if (ico) ico.hidden = true;
  transcribeDropZone.querySelectorAll(':scope > p').forEach(p => p.hidden = true);
  let info = document.getElementById('transcribeFileInfo');
  if (!info) {
    info = document.createElement('p');
    info.id = 'transcribeFileInfo';
    info.style.cssText = 'color:#00ff41;margin-top:8px;font-size:12px;';
    transcribeDropZone.appendChild(info);
  }
  info.hidden = false;
  const sizeMB = (file.size / (1024*1024)).toFixed(1);
  info.innerHTML = '> FILE LOADED: <strong>' + escapeHtml(file.name) + '</strong> (' + sizeMB + ' MB)';
  let changeBtn = document.getElementById('transcribeChangeBtn');
  if (!changeBtn) {
    changeBtn = document.createElement('button');
    changeBtn.id = 'transcribeChangeBtn';
    changeBtn.className = 'change-btn';
    changeBtn.textContent = '[ Change File ]';
    changeBtn.onclick = function(e) { e.stopPropagation(); transcribeFileInput.click(); };
    transcribeDropZone.appendChild(changeBtn);
  }
  changeBtn.hidden = false;
  transcribeDropZone.classList.add('has-image');
  updateTranscribeBtn();
}

function updateTranscribeBtn() {
  const btn = document.getElementById('transcribeBtn');
  if (btn) btn.disabled = !transcribeUploadedFile;
}

async function startTranscribe() {
  const btn = document.getElementById('transcribeBtn');
  const status = document.getElementById('transcribeStatus');
  const results = document.getElementById('transcribeResults');
  btn.disabled = true;
  status.hidden = false;
  document.getElementById('transcribeStatusText').textContent = 'Uploading audio...';
  document.getElementById('transcribeProgressFill').style.width = '0%';
  document.getElementById('transcribeTimerDisplay').textContent = '0:00.0';
  results.innerHTML = '';
  startAudioTimer('transcribe');

  const fd = new FormData();
  fd.append('audio', transcribeUploadedFile);
  fd.append('scan_type', 'transcribe');
  const showTimestamps = document.getElementById('transcribeTimestamps').checked;
  const searchText = document.getElementById('transcribeSearchText').value.trim();
  if (searchText) fd.append('search_text', searchText);

  try {
    const startResp = await fetch('/audio/scan', { method: 'POST', body: fd });
    const startData = await startResp.json();
    if (startData.error) {
      stopAudioTimer(); status.hidden = true;
      results.innerHTML = '<div class="empty-state"><p style="color:#ff6e41;">' + escapeHtml(startData.error) + '</p></div>';
      btn.disabled = false; return;
    }
    const scanId = startData.scan_id;
    document.getElementById('transcribeStatusText').textContent = 'Transcribing...';

    // Live transcript container for streaming
    results.innerHTML = '<div id="transcribeLiveMatches"></div><div id="transcribeLiveSegments" style="margin-top:10px;border:1px solid #0a3e1a;border-radius:6px;padding:12px;background:rgba(0,10,0,0.2);"></div>';

    const data = await new Promise((resolve, reject) => {
      const poll = async () => {
        try {
          const resp = await fetch('/audio/scan/progress?id=' + scanId);
          const prog = await resp.json();
          // Render new partial results live
          if (prog.new_matches && prog.new_matches.length > 0) {
            const matchBox = document.getElementById('transcribeLiveMatches');
            const segBox = document.getElementById('transcribeLiveSegments');
            prog.new_matches.forEach(m => {
              if (m.kind === 'match' && matchBox) {
                const div = document.createElement('div');
                div.className = 'transcript-segment';
                div.style.animation = 'fadeIn 0.3s ease-in';
                div.innerHTML = (showTimestamps ? '<div class="seg-time">' + fmtTime(m.start) + ' — ' + fmtTime(m.end) + '</div>' : '') +
                  '<div class="seg-text">' + m.highlighted_text + '</div>';
                matchBox.appendChild(div);
              } else if (m.kind === 'segment' && segBox) {
                const div = document.createElement('div');
                div.style.cssText = 'margin-bottom:6px;line-height:1.5;animation:fadeIn 0.2s ease-in;';
                div.innerHTML = (showTimestamps ? '<span style="color:#7fff00;font-size:11px;margin-right:6px;">[' + fmtTime(m.start) + ']</span> ' : '') +
                  escapeHtml(m.text);
                segBox.appendChild(div);
              }
            });
          }
          if (prog.done) {
            if (prog.error) { reject(new Error(prog.error)); return; }
            document.getElementById('transcribeProgressFill').style.width = '100%';
            resolve(prog); return;
          }
          document.getElementById('transcribeProgressFill').style.width = prog.percent + '%';
          document.getElementById('transcribeStatusText').textContent = prog.phase_detail || 'Processing...';
          setTimeout(poll, 400);
        } catch (err) { reject(err); }
      };
      poll();
    });

    stopAudioTimer(); status.hidden = true;
    renderTranscribeResults(data, searchText, showTimestamps);
  } catch (err) {
    stopAudioTimer(); status.hidden = true;
    results.innerHTML = '<div class="empty-state"><p style="color:#ff6e41;">Error: ' + escapeHtml(err.message) + '</p></div>';
  }
  btn.disabled = false;
  updateTranscribeBtn();
}

// Store last transcription data for download
let lastTranscriptData = null;
let lastTranscriptTimestamps = true;

function renderTranscribeResults(data, searchText, showTimestamps) {
  const results = document.getElementById('transcribeResults');
  const matches = data.matches || [];
  const lang = data.language ? data.language.toUpperCase() : '??';
  const dur = data.audio_duration ? fmtTime(data.audio_duration) : '??';
  const elapsed = data.elapsed || '?';
  const totalSegs = data.total_segments || 0;

  // Store for download
  lastTranscriptData = data;
  lastTranscriptTimestamps = showTimestamps;

  let html = '<div class="results-header" style="margin-bottom:15px;padding:10px;border:1px solid #0a3e1a;border-radius:4px;color:#0a8e3a;font-size:13px;">';
  html += '> Transcribed <span style="color:#00ff41;">' + totalSegs + '</span> segment(s)';
  html += ' &bull; ' + elapsed + 's &bull; ' + lang + ' &bull; Duration: ' + dur;
  if (searchText && matches.length > 0) {
    html += ' &bull; <span style="color:#00ff41;">' + matches.length + '</span> keyword match(es)';
  }
  html += '</div>';

  // Download buttons
  html += '<div style="margin-bottom:15px;display:flex;gap:8px;flex-wrap:wrap;">';
  html += '<button class="scan-btn" style="font-size:11px;padding:6px 14px;" onclick="downloadTranscript(\'txt\')">[ Download .txt ]</button>';
  html += '<button class="scan-btn" style="font-size:11px;padding:6px 14px;" onclick="downloadTranscript(\'srt\')">[ Download .srt ]</button>';
  html += '<button class="scan-btn" style="font-size:11px;padding:6px 14px;" onclick="downloadTranscript(\'vtt\')">[ Download .vtt ]</button>';
  html += '</div>';

  // Show keyword matches first if any
  if (searchText && matches.length > 0) {
    html += '<div style="margin-bottom:20px;border:1px solid #0a5e2a;border-radius:6px;padding:12px;background:rgba(0,10,0,0.3);">';
    html += '<div style="margin-bottom:10px;font-size:13px;color:#00ff41;"><strong>Keyword Matches for "' + escapeHtml(searchText) + '"</strong></div>';
    for (const m of matches) {
      html += '<div class="transcript-segment">';
      if (showTimestamps) html += '<div class="seg-time">' + fmtTime(m.start) + ' — ' + fmtTime(m.end) + '</div>';
      html += '<div class="seg-text">' + m.highlighted_text + '</div></div>';
    }
    html += '</div>';
  } else if (searchText && matches.length === 0) {
    html += '<div style="margin-bottom:15px;padding:8px 12px;border:1px solid #5a2a0a;border-radius:4px;color:#ff6e41;font-size:12px;">';
    html += '> No matches found for "' + escapeHtml(searchText) + '"</div>';
  }

  // Always show full transcript prominently
  if (data.full_transcript && data.full_transcript.length > 0) {
    html += '<div style="border:1px solid #0a3e1a;border-radius:6px;padding:15px;background:rgba(0,10,0,0.2);">';
    html += '<div style="margin-bottom:12px;font-size:13px;color:#00ff41;"><strong>> Full Transcript</strong></div>';
    for (const seg of data.full_transcript) {
      html += '<div style="margin-bottom:8px;line-height:1.5;">';
      if (showTimestamps) html += '<span style="color:#7fff00;font-size:11px;margin-right:6px;">[' + fmtTime(seg.start) + ']</span> ';
      html += escapeHtml(seg.text) + '</div>';
    }
    html += '</div>';
  } else {
    html += '<div class="empty-state"><p>> No speech detected in the audio.</p></div>';
  }

  results.innerHTML = html;
}

function fmtSrtTime(seconds) {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  const ms = Math.round((seconds % 1) * 1000);
  return String(h).padStart(2,'0') + ':' + String(m).padStart(2,'0') + ':' + String(s).padStart(2,'0') + ',' + String(ms).padStart(3,'0');
}

function fmtVttTime(seconds) {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);
  const ms = Math.round((seconds % 1) * 1000);
  return String(h).padStart(2,'0') + ':' + String(m).padStart(2,'0') + ':' + String(s).padStart(2,'0') + '.' + String(ms).padStart(3,'0');
}

function downloadTranscript(format) {
  if (!lastTranscriptData || !lastTranscriptData.full_transcript) return;
  const segs = lastTranscriptData.full_transcript;
  let content = '';
  let filename = 'transcript';
  let mime = 'text/plain';

  if (format === 'txt') {
    for (const seg of segs) {
      if (lastTranscriptTimestamps) {
        content += '[' + fmtTime(seg.start) + '] ' + seg.text + '\n';
      } else {
        content += seg.text + '\n';
      }
    }
    filename += '.txt';
  } else if (format === 'srt') {
    segs.forEach((seg, i) => {
      content += (i + 1) + '\n';
      content += fmtSrtTime(seg.start) + ' --> ' + fmtSrtTime(seg.end) + '\n';
      content += seg.text + '\n\n';
    });
    filename += '.srt';
    mime = 'text/srt';
  } else if (format === 'vtt') {
    content = 'WEBVTT\n\n';
    segs.forEach((seg, i) => {
      content += (i + 1) + '\n';
      content += fmtVttTime(seg.start) + ' --> ' + fmtVttTime(seg.end) + '\n';
      content += seg.text + '\n\n';
    });
    filename += '.vtt';
    mime = 'text/vtt';
  }

  const blob = new Blob([content], {type: mime});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

// ==================== SCAN HISTORY (text/audio/transcribe) ====================
const scanHistoryOpen = {text: false, audio: false, transcribe: false};

async function toggleScanHistory(type) {
  const panel = document.getElementById(type + 'HistoryPanel');
  const btn = document.getElementById(type + 'HistoryToggle');
  scanHistoryOpen[type] = !scanHistoryOpen[type];
  if (scanHistoryOpen[type]) {
    panel.hidden = false;
    btn.innerHTML = '> Scan History [ Click to collapse ]';
    await loadScanHistory(type);
  } else {
    panel.hidden = true;
    btn.innerHTML = '> Scan History [ Click to expand ]';
  }
}

async function loadScanHistory(type) {
  const list = document.getElementById(type + 'HistoryList');
  const controls = document.getElementById(type + 'HistoryControls');
  const countEl = document.getElementById(type + 'HistoryCount');
  try {
    const resp = await fetch('/scan-history?type=' + type);
    const data = await resp.json();
    if (!data.items || data.items.length === 0) {
      list.innerHTML = '<p style="color:#0a4e2a;text-align:center;padding:15px;">No scan history yet.</p>';
      controls.hidden = true;
      return;
    }
    controls.hidden = false;
    countEl.textContent = data.items.length + ' scan' + (data.items.length !== 1 ? 's' : '');
    let html = '';
    for (const item of data.items) {
      const dur = item.duration ? fmtTime(item.duration) : '??';
      const lang = item.language ? item.language.toUpperCase() : '';
      html += '<div class="scan-history-item" id="shi_' + item.id + '">';
      html += '<div style="display:flex;justify-content:space-between;align-items:center;">';
      html += '<span style="color:#00ff41;font-size:12px;font-weight:bold;">' + escapeHtml(item.filename) + '</span>';
      html += '<span class="hist-delete" style="display:inline;position:static;width:auto;height:auto;line-height:normal;padding:2px 6px;" onclick="deleteScanHistoryItem(\'' + item.id + '\',\'' + type + '\')">&times;</span>';
      html += '</div>';
      html += '<div style="font-size:11px;color:#0a5e2a;margin-top:4px;">';
      html += item.date + ' &bull; ' + dur;
      if (lang) html += ' &bull; ' + lang;
      html += ' &bull; ' + (item.elapsed || '?') + 's';
      if (item.search_text) html += ' &bull; Search: "' + escapeHtml(item.search_text) + '"';
      html += ' &bull; ' + (item.matches_count || 0) + ' match(es)';
      html += '</div>';
      if (item.transcript_preview) {
        html += '<div style="font-size:10px;color:#0a4e2a;margin-top:4px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">' + escapeHtml(item.transcript_preview) + '</div>';
      }
      html += '</div>';
    }
    list.innerHTML = html;
  } catch (err) {
    list.innerHTML = '<p style="color:#ff6e41;text-align:center;padding:15px;">Error loading history</p>';
  }
}

async function deleteScanHistoryItem(id, type) {
  const el = document.getElementById('shi_' + id);
  if (el) el.style.opacity = '0.3';
  try {
    await fetch('/scan-history/delete', { method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({id: id}) });
    if (el) el.remove();
    const list = document.getElementById(type + 'HistoryList');
    const remaining = list.querySelectorAll('.scan-history-item').length;
    const countEl = document.getElementById(type + 'HistoryCount');
    if (remaining === 0) {
      list.innerHTML = '<p style="color:#0a4e2a;text-align:center;padding:15px;">No scan history yet.</p>';
      document.getElementById(type + 'HistoryControls').hidden = true;
    } else {
      countEl.textContent = remaining + ' scan' + (remaining !== 1 ? 's' : '');
    }
  } catch (err) {
    if (el) el.style.opacity = '1';
  }
}

async function clearScanHistory(type) {
  if (!confirm('Delete all scan history? This cannot be undone.')) return;
  try {
    await fetch('/scan-history/clear', { method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({type: type}) });
    await loadScanHistory(type);
  } catch (err) {}
}

// ---- Eye tracking: logo eye follows the mouse ----
(function() {
  const logo = document.getElementById('logoSvg');
  const iris = document.getElementById('eyeIris');
  const pupil = document.getElementById('eyePupil');
  const glint = document.getElementById('eyeGlint');
  if (!logo || !iris || !pupil || !glint) return;

  const CENTER_X = 40, CENTER_Y = 40, MAX_MOVE = 6;

  document.addEventListener('mousemove', function(e) {
    const rect = logo.getBoundingClientRect();
    const logoCX = rect.left + rect.width / 2;
    const logoCY = rect.top + rect.height / 2;

    const dx = e.clientX - logoCX;
    const dy = e.clientY - logoCY;
    const dist = Math.sqrt(dx * dx + dy * dy) || 1;

    // Clamp movement to MAX_MOVE units in SVG space
    const move = Math.min(dist / 15, MAX_MOVE);
    const offsetX = (dx / dist) * move;
    const offsetY = (dy / dist) * move;

    const ix = CENTER_X + offsetX;
    const iy = CENTER_Y + offsetY;

    iris.setAttribute('cx', ix);
    iris.setAttribute('cy', iy);
    pupil.setAttribute('cx', ix);
    pupil.setAttribute('cy', iy);
    glint.setAttribute('cx', ix - 3);
    glint.setAttribute('cy', iy - 3);
  });
})();

</script>
</body>
</html>'''


LOGIN_PAGE = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SEARCH SCANNER // ACCESS</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');
  * { box-sizing: border-box; margin: 0; padding: 0;
      cursor: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='28' height='18' viewBox='0 0 28 18'%3E%3Cpath d='M1 9 Q14 0 27 9 Q14 18 1 9 Z' fill='none' stroke='%2300ff41' stroke-width='1.5'/%3E%3Ccircle cx='14' cy='9' r='4' fill='none' stroke='%2300ff41' stroke-width='1.2'/%3E%3Ccircle cx='14' cy='9' r='1.5' fill='%2300ff41'/%3E%3C/svg%3E") 14 9, auto; }
  body { font-family: 'Share Tech Mono', 'Courier New', monospace;
         background: #000a00; color: #00ff41; min-height: 100vh; display: flex;
         align-items: center; justify-content: center; position: relative; overflow: hidden; }
  #matrixCanvas { position: fixed; top: 0; left: 0; width: 100%; height: 100%;
                  z-index: 0; pointer-events: none; opacity: 0.15; }
  .login-wrap { position: relative; z-index: 1; }
  .login-wrap::before { content: ''; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    background: repeating-linear-gradient(0deg, rgba(0,0,0,0.15) 0px, rgba(0,0,0,0.15) 1px, transparent 1px, transparent 3px);
    pointer-events: none; z-index: 9999; }
  .login-box { background: rgba(0, 10, 0, 0.9); border: 2px solid #00ff41; border-radius: 8px;
    padding: 40px 35px; width: 380px; max-width: 90vw; text-align: center;
    box-shadow: 0 0 40px rgba(0,255,65,0.15), 0 0 80px rgba(0,255,65,0.05);
    backdrop-filter: blur(8px); }
  .login-box .logo-svg { width: 64px; height: 64px; margin-bottom: 15px;
    filter: drop-shadow(0 0 10px #00ff41); }
  .login-box h1 { font-size: 20px; letter-spacing: 3px; text-transform: uppercase;
    text-shadow: 0 0 10px #00ff41, 0 0 30px #008f11; margin-bottom: 6px; }
  .login-box .subtitle { color: #0a6e2a; font-size: 12px; margin-bottom: 28px; letter-spacing: 1px; }
  .login-box label { display: block; text-align: left; color: #0a8e3a; font-size: 11px;
    text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
  .login-box input[type="password"] { width: 100%; padding: 12px 14px; border-radius: 4px;
    border: 1px solid #0a6e2a; background: rgba(0, 5, 0, 0.9); color: #00ff41;
    font-size: 18px; font-family: 'Share Tech Mono', monospace; text-align: center;
    letter-spacing: 6px; margin-bottom: 20px; }
  .login-box input[type="password"]:focus { outline: none; border-color: #00ff41;
    box-shadow: 0 0 15px rgba(0,255,65,0.3); }
  .login-box input[type="password"]::placeholder { color: #0a3e1a; letter-spacing: 2px; font-size: 13px; }
  .login-btn { width: 100%; padding: 14px; background: transparent; color: #00ff41;
    border: 2px solid #00ff41; border-radius: 4px; font-size: 14px; font-weight: 600;
    cursor: pointer; text-transform: uppercase; letter-spacing: 3px;
    font-family: 'Share Tech Mono', monospace; transition: all 0.3s; }
  .login-btn:hover { background: #00ff41; color: #000a00;
    box-shadow: 0 0 20px #00ff41, 0 0 40px rgba(0,255,65,0.3); }
  .error-msg { color: #ff6e41; font-size: 12px; margin-top: 14px; min-height: 16px;
    text-shadow: 0 0 6px rgba(255,110,65,0.5); }
  @keyframes flicker { 0%, 95% { opacity: 1; } 96% { opacity: 0.8; } 97% { opacity: 1; }
    98% { opacity: 0.6; } 100% { opacity: 1; } }
  .login-box h1 { animation: flicker 4s infinite; }
  .lock-icon { font-size: 10px; color: #0a5e2a; margin-top: 18px; }
</style>
</head>
<body>
<canvas id="matrixCanvas"></canvas>
<div class="login-wrap">
<div class="login-box">
  <svg class="logo-svg" id="loginLogoSvg" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
    <circle cx="40" cy="40" r="28" fill="none" stroke="#00ff41" stroke-width="4"/>
    <line x1="60" y1="60" x2="88" y2="88" stroke="#00ff41" stroke-width="6" stroke-linecap="round"/>
    <path d="M22 40 Q40 25 58 40 Q40 55 22 40 Z" fill="none" stroke="#00ff41" stroke-width="2.5"/>
    <circle id="loginIris" cx="40" cy="40" r="7" fill="none" stroke="#00ff41" stroke-width="2"/>
    <circle id="loginPupil" cx="40" cy="40" r="3" fill="#00ff41"/>
    <circle id="loginGlint" cx="37" cy="37" r="1.5" fill="#aaffaa"/>
  </svg>
  <h1>SEARCH SCANNER</h1>
  <p class="subtitle">> authentication required _</p>
  <form method="POST" action="/login" id="loginForm">
    <label>> Enter Access Code</label>
    <input type="password" name="password" id="passInput" placeholder="****" autofocus autocomplete="off">
    <button type="submit" class="login-btn">[ Authenticate ]</button>
  </form>
  <div class="error-msg" id="errorMsg">''' + '{{ error }}' + r'''</div>
  <div class="lock-icon">&#128274; ENCRYPTED SESSION</div>
</div>
</div>
<script>
const canvas = document.getElementById('matrixCanvas');
const ctx = canvas.getContext('2d');
canvas.width = window.innerWidth;
canvas.height = window.innerHeight;
window.addEventListener('resize', () => { canvas.width = window.innerWidth; canvas.height = window.innerHeight; });
const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789@#$%^&*(){}[]|;:<>,.?/~`';
const fontSize = 14;
const columns = Math.floor(canvas.width / fontSize);
const drops = Array(columns).fill(1);
function drawMatrix() {
  ctx.fillStyle = 'rgba(0, 10, 0, 0.05)';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#00ff41';
  ctx.font = fontSize + 'px monospace';
  for (let i = 0; i < drops.length; i++) {
    const text = chars[Math.floor(Math.random() * chars.length)];
    ctx.fillText(text, i * fontSize, drops[i] * fontSize);
    if (drops[i] * fontSize > canvas.height && Math.random() > 0.975) drops[i] = 0;
    drops[i]++;
  }
}
setInterval(drawMatrix, 50);

// Eye tracking for login page logo
(function() {
  const logo = document.getElementById('loginLogoSvg');
  const iris = document.getElementById('loginIris');
  const pupil = document.getElementById('loginPupil');
  const glint = document.getElementById('loginGlint');
  if (!logo || !iris || !pupil || !glint) return;
  const CX = 40, CY = 40, MAX = 6;
  document.addEventListener('mousemove', function(e) {
    const r = logo.getBoundingClientRect();
    const dx = e.clientX - (r.left + r.width/2);
    const dy = e.clientY - (r.top + r.height/2);
    const d = Math.sqrt(dx*dx + dy*dy) || 1;
    const m = Math.min(d / 15, MAX);
    const ox = (dx/d)*m, oy = (dy/d)*m;
    iris.setAttribute('cx', CX+ox); iris.setAttribute('cy', CY+oy);
    pupil.setAttribute('cx', CX+ox); pupil.setAttribute('cy', CY+oy);
    glint.setAttribute('cx', CX+ox-3); glint.setAttribute('cy', CY+oy-3);
  });
})();
</script>
</body>
</html>'''


@app.before_request
def require_login():
    """Require authentication for all routes except /login. Enforce session timeout."""
    if request.endpoint in ('login', 'static'):
        return
    if not session.get('authenticated'):
        if request.is_json or request.headers.get('X-Requested-With'):
            return jsonify(error="Authentication required"), 401
        return redirect(url_for('login'))
    # Session timeout check (1 hour inactivity)
    last_active = session.get('last_active', 0)
    if time.time() - last_active > app.config['PERMANENT_SESSION_LIFETIME']:
        session.clear()
        _sec_log.info(f"Session timed out for {_get_client_ip()}")
        return redirect(url_for('login'))
    session['last_active'] = time.time()


@app.after_request
def add_security_headers(response):
    """Add security headers to every response."""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, private'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Permissions-Policy'] = 'camera=(), microphone=(), geolocation=()'
    # Content Security Policy — allow inline styles/scripts (needed for embedded app), block everything else
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src 'self' https://fonts.gstatic.com; "
        "img-src 'self' data:; "
        "connect-src 'self'; "
        "frame-ancestors 'none'; "
        "base-uri 'self'; "
        "form-action 'self';"
    )
    return response


@app.route('/login', methods=['GET', 'POST'])
def login():
    ip = _get_client_ip()
    error = ''

    # Check if IP is locked out
    if _is_ip_locked(ip):
        remaining = int(_locked_ips.get(ip, 0) - time.time())
        error = f'> LOCKED OUT — try again in {remaining}s'
        _sec_log.warning(f"Locked IP {ip} attempted login")
        return LOGIN_PAGE.replace('{{ error }}', error), 429

    if request.method == 'POST':
        pw = request.form.get('password', '')
        # Rate limit: add small delay to slow automated attacks
        time.sleep(0.5)
        if _verify_password(pw):
            _clear_login_attempts(ip)
            session.clear()  # Regenerate session on login (session fixation protection)
            session['authenticated'] = True
            session['last_active'] = time.time()
            session['login_ip'] = ip
            session.permanent = True
            _sec_log.info(f"Successful login from {ip}")
            return redirect(url_for('index'))
        else:
            locked = _record_failed_login(ip)
            attempts_left = MAX_LOGIN_ATTEMPTS - len(_login_attempts.get(ip, []))
            _sec_log.warning(f"Failed login from {ip} ({attempts_left} attempts left)")
            if locked:
                error = f'> LOCKED OUT — too many attempts. Wait {LOCKOUT_DURATION // 60} min'
            else:
                error = f'> ACCESS DENIED — invalid code ({attempts_left} attempts left)'
    return LOGIN_PAGE.replace('{{ error }}', error)


@app.route('/logout')
def logout():
    ip = _get_client_ip()
    _sec_log.info(f"Logout from {ip}")
    session.clear()
    return redirect(url_for('login'))


@app.route('/')
def index():
    return HTML_PAGE


@app.route('/scan', methods=['POST'])
def scan():
    """Start a scan in a background thread and return scan_id for progress polling."""
    if 'image' not in request.files:
        return jsonify(error="No image uploaded"), 400

    file = request.files['image']

    # Validate uploaded file (extension + magic bytes)
    valid, err_msg = _validate_upload_file(file)
    if not valid:
        return jsonify(error=err_msg), 400

    folder = request.form.get('folder', '').strip()
    try:
        threshold = int(request.form.get('threshold', DEFAULT_THRESHOLD))
        threshold = max(0, min(threshold, 64))  # Clamp to valid range
    except (ValueError, TypeError):
        threshold = DEFAULT_THRESHOLD

    # Sanitize folder path
    safe_folder = _safe_path(folder)
    if not safe_folder or not os.path.isdir(safe_folder):
        return jsonify(error="Folder not found or access denied"), 400

    # Use only safe extension from whitelist
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_UPLOAD_EXTENSIONS:
        ext = '.png'
    temp_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}{ext}")
    file.save(temp_path)

    # Save to history
    try:
        import shutil
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        safe_name = "".join(c if c.isalnum() or c in '.-_' else '_' for c in file.filename)[:100]
        history_name = f"{timestamp}_{safe_name}"
        history_path = os.path.join(HISTORY_DIR, history_name)
        shutil.copy2(temp_path, history_path)
    except Exception:
        pass

    scan_id = uuid.uuid4().hex[:12]

    # Initialize progress
    with _scan_lock:
        _scan_progress[scan_id] = {
            'processed': 0, 'total': 0, 'matches': 0,
            'total_images': 0, 'total_videos': 0,
            'done': False, 'error': None, 'result': None,
            'start_time': time.time(),
            'partial_results': [],
            'partial_sent': 0,
        }

    def run_scan():
        try:
            scanner = ImageScanner(temp_path, safe_folder, threshold)
            results, errors, total_images, total_videos = scanner.scan(scan_id=scan_id)
            elapsed = time.time() - _scan_progress[scan_id]['start_time']

            result_data = []
            for path, distance, media_type, extra in results:
                if media_type == 'video' and extra:
                    result_data.append({
                        'type': 'video',
                        'path': path,
                        'folder': os.path.dirname(path),
                        'distance': distance,
                        'timestamp': extra['timestamp'],
                        'thumbnail': make_thumbnail_b64(frame_image=extra['frame']),
                    })
                else:
                    result_data.append({
                        'type': 'image',
                        'path': path,
                        'folder': os.path.dirname(path),
                        'distance': distance,
                        'thumbnail': make_thumbnail_b64(image_path=path),
                    })

            with _scan_lock:
                _scan_progress[scan_id]['done'] = True
                _scan_progress[scan_id]['result'] = {
                    'results': result_data, 'elapsed': elapsed,
                    'total_images_scanned': total_images,
                    'total_videos_scanned': total_videos,
                    'errors': errors,
                }
        except Exception as e:
            with _scan_lock:
                _scan_progress[scan_id]['done'] = True
                _scan_progress[scan_id]['error'] = str(e)
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    threading.Thread(target=run_scan, daemon=True).start()
    return jsonify(scan_id=scan_id)


@app.route('/scan/progress')
def scan_progress():
    """Poll scan progress. Returns processed/total/matches or final results when done."""
    scan_id = request.args.get('id', '')
    with _scan_lock:
        prog = _scan_progress.get(scan_id)
    if not prog:
        return jsonify(error="Unknown scan"), 404

    if prog['done']:
        # Return final results and clean up
        with _scan_lock:
            result = _scan_progress.pop(scan_id, {})
        if result.get('error'):
            return jsonify(error=result['error']), 500
        return jsonify(done=True, **result['result'])
    else:
        # Send only NEW partial results since last poll
        new_results = []
        with _scan_lock:
            sent = prog.get('partial_sent', 0)
            all_partial = prog.get('partial_results', [])
            if sent < len(all_partial):
                new_results = all_partial[sent:]
                _scan_progress[scan_id]['partial_sent'] = len(all_partial)
        return jsonify(
            done=False,
            processed=prog['processed'],
            total=prog['total'],
            matches=prog['matches'],
            total_images=prog['total_images'],
            total_videos=prog['total_videos'],
            new_matches=new_results,
        )


@app.route('/audio/status')
def audio_status():
    """Check if audio scanning is available."""
    return jsonify(available=HAS_WHISPER)


@app.route('/audio/scan', methods=['POST'])
def audio_scan():
    """Start audio transcription + keyword search (search_text is optional for transcribe-only mode)."""
    if not HAS_WHISPER:
        return jsonify(error="Audio scanning not available. Install faster-whisper."), 501
    if 'audio' not in request.files:
        return jsonify(error="No audio file uploaded"), 400
    file = request.files['audio']
    search_text = request.form.get('search_text', '').strip()
    scan_type = request.form.get('scan_type', 'text')  # 'text' or 'transcribe'
    original_filename = file.filename or 'unknown'
    valid, err_msg = _validate_audio_upload(file)
    if not valid:
        return jsonify(error=err_msg), 400
    ext = Path(file.filename).suffix.lower()
    is_video = ext in VIDEO_EXTENSIONS
    if ext not in AUDIO_EXTENSIONS and not is_video:
        ext = '.wav'
    temp_path = os.path.join(AUDIO_UPLOAD_DIR, f"{uuid.uuid4().hex}{ext}")
    file.save(temp_path)
    scan_id = uuid.uuid4().hex[:12]
    with _audio_scan_lock:
        _audio_scan_progress[scan_id] = {
            'phase': 'starting', 'phase_detail': 'Initializing...',
            'percent': 0, 'done': False, 'error': None, 'result': None,
            'start_time': time.time(),
            'partial_results': [],
            'partial_sent': 0,
        }

    def run_audio_scan():
        audio_path = temp_path
        extracted_path = None
        try:
            # If video file, extract audio first
            if is_video:
                with _audio_scan_lock:
                    _audio_scan_progress[scan_id].update({
                        'phase': 'extracting', 'phase_detail': 'Extracting audio from video...',
                        'percent': 2,
                    })
                extracted_path = AudioScanner.extract_audio_from_video(temp_path)
                if not extracted_path:
                    raise Exception("Could not extract audio from video file")
                audio_path = extracted_path
            scanner = AudioScanner(audio_path, search_text)
            result = scanner.transcribe_and_search(scan_id=scan_id)
            elapsed = time.time() - _audio_scan_progress[scan_id]['start_time']
            result['elapsed'] = round(elapsed, 1)
            result['search_text'] = search_text
            with _audio_scan_lock:
                _audio_scan_progress[scan_id]['done'] = True
                _audio_scan_progress[scan_id]['result'] = result
                _audio_scan_progress[scan_id]['percent'] = 100
            # Save to scan history
            transcript_text = ' '.join(s['text'] for s in (result.get('full_transcript') or []))
            _add_scan_history({
                'type': scan_type,
                'filename': original_filename,
                'duration': result.get('audio_duration', 0),
                'language': result.get('language', ''),
                'search_text': search_text,
                'matches_count': len(result.get('matches', [])),
                'segments_count': result.get('total_segments', 0),
                'transcript_preview': transcript_text[:200],
                'elapsed': result.get('elapsed', 0),
            })
        except Exception as e:
            with _audio_scan_lock:
                _audio_scan_progress[scan_id]['done'] = True
                _audio_scan_progress[scan_id]['error'] = str(e)
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass
            if extracted_path:
                try:
                    os.unlink(extracted_path)
                except OSError:
                    pass

    threading.Thread(target=run_audio_scan, daemon=True).start()
    return jsonify(scan_id=scan_id)


@app.route('/audio/scan/progress')
def audio_scan_progress():
    """Poll audio scan progress. Returns new_matches for live streaming."""
    scan_id = request.args.get('id', '')
    with _audio_scan_lock:
        prog = _audio_scan_progress.get(scan_id)
    if not prog:
        return jsonify(error="Unknown scan"), 404
    if prog['done']:
        with _audio_scan_lock:
            result = _audio_scan_progress.pop(scan_id, {})
        if result.get('error'):
            return jsonify(error=result['error'], done=True), 500
        return jsonify(done=True, **result['result'])
    # Send only NEW partial results since last poll
    new_results = []
    with _audio_scan_lock:
        sent = prog.get('partial_sent', 0)
        all_partial = prog.get('partial_results', [])
        if sent < len(all_partial):
            new_results = all_partial[sent:]
            _audio_scan_progress[scan_id]['partial_sent'] = len(all_partial)
    return jsonify(
        done=False, phase=prog['phase'],
        phase_detail=prog['phase_detail'], percent=prog['percent'],
        new_matches=new_results,
    )


@app.route('/audio/folder-scan', methods=['POST'])
def audio_folder_scan():
    """Scan all audio & video files in a folder for keyword matches."""
    if not HAS_WHISPER:
        return jsonify(error="Audio scanning not available. Install faster-whisper."), 501
    data = request.get_json()
    folder = data.get('folder', '')
    search_text = data.get('search_text', '').strip()
    if not search_text:
        return jsonify(error="No search text provided"), 400
    safe = _safe_path(folder)
    if not safe or not os.path.isdir(safe):
        return jsonify(error="Invalid folder"), 400

    scan_id = uuid.uuid4().hex[:12]
    with _audio_scan_lock:
        _audio_scan_progress[scan_id] = {
            'phase': 'collecting', 'phase_detail': 'Scanning folder for audio/video files...',
            'percent': 0, 'done': False, 'error': None, 'result': None,
            'start_time': time.time(),
            'partial_results': [],
            'partial_sent': 0,
        }

    def run_folder_scan():
        temp_files = []
        try:
            files = AudioScanner.collect_audio_files(safe)
            total = len(files)
            if total == 0:
                with _audio_scan_lock:
                    _audio_scan_progress[scan_id]['done'] = True
                    _audio_scan_progress[scan_id]['result'] = {
                        'file_results': [], 'total_files': 0,
                        'total_matches': 0, 'search_text': search_text,
                        'elapsed': round(time.time() - _audio_scan_progress[scan_id]['start_time'], 1),
                    }
                return

            with _audio_scan_lock:
                _audio_scan_progress[scan_id].update({
                    'phase': 'scanning',
                    'phase_detail': f'Found {total} audio/video file(s). Starting transcription...',
                    'percent': 2,
                })

            all_results = []
            total_matches = 0

            for i, fpath in enumerate(files):
                fname = os.path.basename(fpath)
                ext = Path(fpath).suffix.lower()
                is_video = ext in VIDEO_EXTENSIONS

                # For video files, extract audio first
                audio_path = fpath
                if is_video:
                    with _audio_scan_lock:
                        _audio_scan_progress[scan_id].update({
                            'phase': 'extracting',
                            'phase_detail': f'[{i+1}/{total}] Extracting audio from {fname}...',
                            'percent': int(5 + 90 * i / total),
                        })
                    extracted = AudioScanner.extract_audio_from_video(fpath)
                    if not extracted:
                        all_results.append({
                            'file': fpath, 'filename': fname, 'is_video': True,
                            'error': 'Could not extract audio', 'matches': [],
                            'full_transcript': [], 'audio_duration': 0,
                        })
                        continue
                    audio_path = extracted
                    temp_files.append(extracted)

                pct_base = int(5 + 90 * i / total)
                pct_range = int(90 / total)

                try:
                    scanner = AudioScanner(audio_path, search_text)
                    result = scanner.transcribe_and_search(
                        scan_id=scan_id,
                        file_label=f'{i+1}/{total} {fname}',
                        pct_base=pct_base, pct_range=pct_range)
                    result['file'] = fpath
                    result['filename'] = fname
                    result['is_video'] = is_video
                    total_matches += len(result['matches'])
                    all_results.append(result)
                    # Push per-file completion for live folder scan display
                    with _audio_scan_lock:
                        if scan_id in _audio_scan_progress:
                            _audio_scan_progress[scan_id]['partial_results'].append({
                                'kind': 'file_done',
                                'filename': fname,
                                'file': fpath,
                                'is_video': is_video,
                                'matches_count': len(result['matches']),
                                'matches': result['matches'][:10],  # first 10 for display
                                'audio_duration': result.get('audio_duration', 0),
                                'language': result.get('language', ''),
                                'total_segments': result.get('total_segments', 0),
                            })
                except Exception as e:
                    all_results.append({
                        'file': fpath, 'filename': fname, 'is_video': is_video,
                        'error': str(e), 'matches': [],
                        'full_transcript': [], 'audio_duration': 0,
                    })

            elapsed = round(time.time() - _audio_scan_progress[scan_id]['start_time'], 1)
            with _audio_scan_lock:
                _audio_scan_progress[scan_id]['done'] = True
                _audio_scan_progress[scan_id]['result'] = {
                    'file_results': all_results,
                    'total_files': total,
                    'total_matches': total_matches,
                    'search_text': search_text,
                    'elapsed': elapsed,
                }
                _audio_scan_progress[scan_id]['percent'] = 100
            # Save to scan history
            _add_scan_history({
                'type': 'audio',
                'filename': os.path.basename(safe),
                'duration': sum(r.get('audio_duration', 0) for r in all_results if not r.get('error')),
                'language': '',
                'search_text': search_text,
                'matches_count': total_matches,
                'segments_count': total,
                'transcript_preview': f'{total} files scanned, {total_matches} matches',
                'elapsed': elapsed,
            })
        except Exception as e:
            with _audio_scan_lock:
                _audio_scan_progress[scan_id]['done'] = True
                _audio_scan_progress[scan_id]['error'] = str(e)
        finally:
            for tf in temp_files:
                try:
                    os.unlink(tf)
                except OSError:
                    pass

    threading.Thread(target=run_folder_scan, daemon=True).start()
    return jsonify(scan_id=scan_id)


@app.route('/reveal', methods=['POST'])
def reveal():
    data = request.get_json()
    path = data.get('path', '')
    safe = _safe_path(path)
    if safe and os.path.exists(safe):
        # Only allow revealing files, not arbitrary command execution
        subprocess.run(['open', '-R', safe], check=False)
        return jsonify(ok=True)
    return jsonify(error="File not found"), 404


@app.route('/browse/list', methods=['POST'])
def browse_list():
    """List subdirectories of a given path for the in-browser folder picker."""
    data = request.get_json()
    raw_path = data.get('path', '~')

    # Sanitize path
    safe = _safe_path(raw_path)
    if not safe or not os.path.isdir(safe):
        return jsonify(error="Not a valid directory")

    parent = os.path.dirname(safe) if safe != '/' else None

    dirs = []
    try:
        entries = sorted(os.listdir(safe), key=lambda s: s.lower())
        for name in entries:
            if name.startswith('.'):
                continue  # Skip hidden
            full = os.path.join(safe, name)
            if os.path.isdir(full):
                dirs.append({'name': name, 'path': full})
    except PermissionError:
        return jsonify(error="Permission denied", current=safe, parent=parent, dirs=[])

    return jsonify(current=safe, parent=parent, dirs=dirs)


@app.route('/history')
def history():
    items = []
    try:
        for fname in sorted(os.listdir(HISTORY_DIR), reverse=True):
            fpath = os.path.join(HISTORY_DIR, fname)
            if Path(fname).suffix.lower() in SUPPORTED_EXTENSIONS and os.path.isfile(fpath):
                thumb = make_thumbnail_b64(fpath)
                items.append({
                    'filename': fname,
                    'path': fpath,
                    'thumbnail': thumb,
                    'size': os.path.getsize(fpath),
                })
            if len(items) >= 50:
                break
    except Exception:
        pass
    return jsonify(items=items, folder=HISTORY_DIR)


@app.route('/history/use', methods=['POST'])
def history_use():
    data = request.get_json()
    path = data.get('path', '')
    safe = _safe_path(path, must_be_under=HISTORY_DIR)
    if safe and os.path.isfile(safe):
        return send_file(safe)
    return jsonify(error="File not found"), 404


@app.route('/history/delete', methods=['POST'])
def history_delete():
    """Delete a single history image."""
    data = request.get_json()
    path = data.get('path', '')
    safe = _safe_path(path, must_be_under=HISTORY_DIR)
    if safe and os.path.isfile(safe):
        try:
            os.unlink(safe)
            _sec_log.info(f"History file deleted by {_get_client_ip()}: {safe}")
            return jsonify(ok=True)
        except Exception as e:
            return jsonify(error="Delete failed"), 500
    return jsonify(error="File not found"), 404


@app.route('/history/clear', methods=['POST'])
def history_clear():
    """Delete all history images."""
    deleted = 0
    try:
        for fname in os.listdir(HISTORY_DIR):
            fpath = os.path.join(HISTORY_DIR, fname)
            if os.path.isfile(fpath):
                os.unlink(fpath)
                deleted += 1
        _sec_log.info(f"All history cleared by {_get_client_ip()}: {deleted} files")
    except Exception as e:
        return jsonify(error="Clear failed"), 500
    return jsonify(ok=True, deleted=deleted)


# --- Scan History Routes (audio/text/transcribe) ---

@app.route('/scan-history')
def scan_history():
    """Get scan history entries, optionally filtered by type."""
    scan_type = request.args.get('type', '')  # 'text', 'audio', 'transcribe' or '' for all
    with _scan_history_lock:
        entries = _load_scan_history()
    if scan_type:
        entries = [e for e in entries if e.get('type') == scan_type]
    entries.reverse()  # newest first
    return jsonify(items=entries[:50])

@app.route('/scan-history/delete', methods=['POST'])
def scan_history_delete():
    """Delete a single scan history entry by id."""
    data = request.get_json()
    entry_id = data.get('id', '')
    with _scan_history_lock:
        entries = _load_scan_history()
        entries = [e for e in entries if e.get('id') != entry_id]
        _save_scan_history(entries)
    return jsonify(ok=True)

@app.route('/scan-history/clear', methods=['POST'])
def scan_history_clear():
    """Clear all scan history entries of a given type."""
    data = request.get_json()
    scan_type = data.get('type', '')
    with _scan_history_lock:
        entries = _load_scan_history()
        if scan_type:
            entries = [e for e in entries if e.get('type') != scan_type]
        else:
            entries = []
        _save_scan_history(entries)
    return jsonify(ok=True)


# --- Error Handlers (prevent info leakage) ---

@app.errorhandler(404)
def not_found(e):
    return jsonify(error="Not found"), 404

@app.errorhandler(413)
def too_large(e):
    return jsonify(error="File too large (500MB max)"), 413

@app.errorhandler(500)
def server_error(e):
    _sec_log.error(f"Internal error from {_get_client_ip()}: {e}")
    return jsonify(error="Internal server error"), 500


def main():
    import webbrowser
    import sys
    port = 8457
    print(f"\n  SEARCH SCANNER")
    print(f"  Security log: ~/.searchscanner/security.log")
    print(f"  Open in browser: http://localhost:{port}\n")
    # Only open browser if launched manually (not by launchd auto-restart)
    if '--no-browser' not in sys.argv:
        webbrowser.open(f"http://localhost:{port}")
    # Listen on all interfaces so other devices on the network can connect
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
