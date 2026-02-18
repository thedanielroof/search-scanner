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

# Algorithm analysis progress tracking
_algorithm_progress = {}
_algorithm_lock = threading.Lock()
ALGORITHM_UPLOAD_DIR = tempfile.mkdtemp(prefix="algorithm_")

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


# ==================== ALGORITHM: Platform Specs ====================
PLATFORM_SPECS = {
    'tiktok': {
        'name': 'TikTok', 'icon': '&#9835;',
        'ideal_aspect': (9, 16), 'alt_aspects': [],
        'aspect_tolerance': 0.15,
        'ideal_duration': (15, 60), 'ok_duration': (5, 180),
        'ideal_resolution': (1080, 1920), 'min_resolution': (720, 1280),
        'ideal_fps': (24, 30),
        'audio_required': True, 'captions_boost': True, 'captions_critical': False,
        'faces_boost': True, 'fast_cuts_boost': True, 'hook_critical': True, 'saturation_boost': False,
        'weights': {'duration': 0.25, 'aspect_ratio': 0.20, 'resolution': 0.12, 'audio': 0.10, 'engagement': 0.23, 'platform_bonus': 0.10},
        'tips': {
            'vertical': 'Use 9:16 vertical format for maximum TikTok reach',
            'duration_short': 'Aim for 15-60 seconds for optimal TikTok distribution',
            'duration_long': 'Trim to under 60s — TikTok favors shorter loops',
            'audio': 'Add trending audio/music — sound is essential on TikTok',
            'hook': 'First 3 seconds must grab attention — add movement or a hook',
            'cuts': 'Add more quick cuts — fast-paced editing performs better',
            'faces': 'Show faces prominently — face content gets 38% more likes',
            'captions': 'Add text overlays — many viewers scroll with sound off initially',
            'brightness': 'Improve lighting — well-lit content gets pushed by the algorithm',
            'motion': 'Add more movement/action — static content gets less distribution',
            'resolution': 'Increase to at least 720x1280 (ideally 1080x1920)',
            'saturation': 'Increase color vibrancy to catch attention in the feed',
        },
    },
    'instagram_reels': {
        'name': 'Instagram Reels', 'icon': '&#128247;',
        'ideal_aspect': (9, 16), 'alt_aspects': [],
        'aspect_tolerance': 0.10,
        'ideal_duration': (15, 30), 'ok_duration': (5, 90),
        'ideal_resolution': (1080, 1920), 'min_resolution': (720, 1280),
        'ideal_fps': (30, 30),
        'audio_required': True, 'captions_boost': True, 'captions_critical': False,
        'faces_boost': True, 'fast_cuts_boost': True, 'hook_critical': True, 'saturation_boost': True,
        'weights': {'duration': 0.25, 'aspect_ratio': 0.20, 'resolution': 0.15, 'audio': 0.10, 'engagement': 0.20, 'platform_bonus': 0.10},
        'tips': {
            'vertical': 'Use 9:16 vertical — Reels are optimized for full-screen vertical',
            'duration_short': 'Keep between 15-30 seconds for maximum Reels reach',
            'duration_long': 'Trim to under 30s — shorter Reels get more replays',
            'audio': 'Add trending audio — Instagram prioritizes Reels with music',
            'hook': 'Grab attention in the first second — start with action or text',
            'cuts': 'Use dynamic transitions — Reels algorithm favors fast pacing',
            'faces': 'Feature faces — Instagram boosts content showing people',
            'captions': 'Add captions — increases watch time and accessibility',
            'brightness': 'Use bright, natural lighting for professional look',
            'motion': 'Add motion and energy — static content underperforms',
            'resolution': 'Upload in 1080x1920 — Instagram rewards high-quality video',
            'saturation': 'Boost color saturation — vibrant visuals stand out on Instagram',
        },
    },
    'youtube_shorts': {
        'name': 'YouTube Shorts', 'icon': '&#9654;',
        'ideal_aspect': (9, 16), 'alt_aspects': [],
        'aspect_tolerance': 0.10,
        'ideal_duration': (15, 58), 'ok_duration': (5, 60),
        'ideal_resolution': (1080, 1920), 'min_resolution': (720, 1280),
        'ideal_fps': (24, 60),
        'audio_required': True, 'captions_boost': True, 'captions_critical': False,
        'faces_boost': True, 'fast_cuts_boost': False, 'hook_critical': True, 'saturation_boost': False,
        'weights': {'duration': 0.25, 'aspect_ratio': 0.22, 'resolution': 0.13, 'audio': 0.10, 'engagement': 0.20, 'platform_bonus': 0.10},
        'tips': {
            'vertical': 'Must be 9:16 vertical to qualify as a Short',
            'duration_short': 'Sweet spot is 15-58 seconds for Shorts',
            'duration_long': 'MUST be under 60 seconds to qualify as a YouTube Short',
            'audio': 'Add clear audio — Shorts with sound perform significantly better',
            'hook': 'Hook viewers in 2-3 seconds — Shorts have fast swipe behavior',
            'cuts': 'Keep pacing engaging but not too frantic',
            'faces': 'Show your face — YouTube promotes creator-focused content',
            'captions': 'Add text overlays for better retention',
            'brightness': 'Good lighting improves viewer retention',
            'motion': 'Keep it visually dynamic — movement holds attention',
            'resolution': 'Upload at 1080x1920 for crisp Shorts',
            'saturation': 'Use vivid colors for thumbnail appeal',
        },
    },
    'youtube': {
        'name': 'YouTube', 'icon': '&#9658;',
        'ideal_aspect': (16, 9), 'alt_aspects': [],
        'aspect_tolerance': 0.10,
        'ideal_duration': (480, 900), 'ok_duration': (60, 3600),
        'ideal_resolution': (1920, 1080), 'min_resolution': (1280, 720),
        'ideal_fps': (24, 60),
        'audio_required': True, 'captions_boost': True, 'captions_critical': False,
        'faces_boost': True, 'fast_cuts_boost': False, 'hook_critical': True, 'saturation_boost': False,
        'weights': {'duration': 0.20, 'aspect_ratio': 0.20, 'resolution': 0.15, 'audio': 0.15, 'engagement': 0.20, 'platform_bonus': 0.10},
        'tips': {
            'vertical': 'Use 16:9 horizontal format for standard YouTube',
            'duration_short': 'YouTube rewards longer watch time — aim for 8-15 minutes',
            'duration_long': 'Consider breaking into segments for better retention',
            'audio': 'High-quality audio is critical — use a good mic',
            'hook': 'First 30 seconds determine if viewers stay — deliver value immediately',
            'cuts': 'Use a natural editing pace — match cuts to content rhythm',
            'faces': 'Talking head content builds audience connection',
            'captions': 'Enable closed captions — improves SEO and accessibility',
            'brightness': 'Professional lighting increases perceived quality',
            'motion': 'Vary shots and angles to maintain visual interest',
            'resolution': 'Upload in 1080p or higher — YouTube promotes HD content',
            'saturation': 'Balanced colors look professional',
        },
    },
    'facebook': {
        'name': 'Facebook', 'icon': '&#402;',
        'ideal_aspect': (1, 1), 'alt_aspects': [(4, 5)],
        'aspect_tolerance': 0.20,
        'ideal_duration': (30, 180), 'ok_duration': (10, 600),
        'ideal_resolution': (1080, 1080), 'min_resolution': (720, 720),
        'ideal_fps': (24, 30),
        'audio_required': False, 'captions_boost': False, 'captions_critical': True,
        'faces_boost': True, 'fast_cuts_boost': False, 'hook_critical': True, 'saturation_boost': False,
        'weights': {'duration': 0.22, 'aspect_ratio': 0.18, 'resolution': 0.12, 'audio': 0.08, 'engagement': 0.20, 'platform_bonus': 0.20},
        'tips': {
            'vertical': 'Use 1:1 square or 4:5 portrait for Facebook feed',
            'duration_short': 'Aim for 30 seconds to 3 minutes for best reach',
            'duration_long': 'Trim to under 3 minutes — Facebook favors concise videos',
            'audio': 'Audio helps but is optional — 85% watch on mute',
            'hook': 'Capture attention in 3 seconds — users scroll fast',
            'cuts': 'Moderate pacing works best for Facebook audiences',
            'faces': 'Videos with faces get 38% higher engagement',
            'captions': 'CAPTIONS ARE CRITICAL — 85% of Facebook videos are watched without sound',
            'brightness': 'Bright, clear video stands out in the feed',
            'motion': 'Add some movement to catch the eye while scrolling',
            'resolution': 'Upload at 1080x1080 minimum',
            'saturation': 'Natural colors work best on Facebook',
        },
    },
    'twitter': {
        'name': 'Twitter / X', 'icon': '&#120143;',
        'ideal_aspect': (16, 9), 'alt_aspects': [(1, 1)],
        'aspect_tolerance': 0.15,
        'ideal_duration': (15, 45), 'ok_duration': (5, 140),
        'ideal_resolution': (1280, 720), 'min_resolution': (640, 360),
        'ideal_fps': (24, 30),
        'audio_required': False, 'captions_boost': False, 'captions_critical': True,
        'faces_boost': False, 'fast_cuts_boost': False, 'hook_critical': True, 'saturation_boost': False,
        'weights': {'duration': 0.25, 'aspect_ratio': 0.18, 'resolution': 0.12, 'audio': 0.08, 'engagement': 0.17, 'platform_bonus': 0.20},
        'tips': {
            'vertical': 'Use 16:9 landscape or 1:1 square for Twitter/X',
            'duration_short': 'Keep to 15-45 seconds — Twitter is fast-paced',
            'duration_long': 'Under 2:20 is the max — shorter is better',
            'audio': 'Audio optional — videos autoplay on mute',
            'hook': 'Start with the most compelling moment',
            'cuts': 'Direct and punchy editing works best',
            'faces': 'Not critical for Twitter but helps engagement',
            'captions': 'ADD CAPTIONS — videos autoplay muted in the feed',
            'brightness': 'Clear visuals stand out in the timeline',
            'motion': 'Movement catches attention while scrolling',
            'resolution': '720p minimum for decent quality',
            'saturation': 'High-contrast colors help in small preview',
        },
    },
    'linkedin': {
        'name': 'LinkedIn', 'icon': '&#128188;',
        'ideal_aspect': (1, 1), 'alt_aspects': [(16, 9)],
        'aspect_tolerance': 0.15,
        'ideal_duration': (30, 300), 'ok_duration': (10, 600),
        'ideal_resolution': (1080, 1080), 'min_resolution': (720, 720),
        'ideal_fps': (24, 30),
        'audio_required': True, 'captions_boost': False, 'captions_critical': True,
        'faces_boost': True, 'fast_cuts_boost': False, 'hook_critical': False, 'saturation_boost': False,
        'weights': {'duration': 0.22, 'aspect_ratio': 0.18, 'resolution': 0.12, 'audio': 0.12, 'engagement': 0.16, 'platform_bonus': 0.20},
        'tips': {
            'vertical': 'Use 1:1 square or 16:9 landscape for professional appearance',
            'duration_short': 'Aim for 30 seconds to 5 minutes for LinkedIn',
            'duration_long': 'Keep professional and concise — respect your audience',
            'audio': 'Clear, professional audio enhances credibility',
            'hook': 'Lead with a strong opening statement or question',
            'cuts': 'Professional pacing — not too fast, not too slow',
            'faces': 'Talking head builds trust and professional presence',
            'captions': 'SUBTITLES ESSENTIAL — most professionals browse on mute',
            'brightness': 'Professional lighting conveys competence',
            'motion': 'Steady, purposeful movement — avoid chaotic editing',
            'resolution': '1080p shows professionalism',
            'saturation': 'Natural, professional color grading',
        },
    },
    'snapchat': {
        'name': 'Snapchat', 'icon': '&#128123;',
        'ideal_aspect': (9, 16), 'alt_aspects': [],
        'aspect_tolerance': 0.10,
        'ideal_duration': (3, 10), 'ok_duration': (1, 60),
        'ideal_resolution': (1080, 1920), 'min_resolution': (720, 1280),
        'ideal_fps': (24, 30),
        'audio_required': True, 'captions_boost': False, 'captions_critical': False,
        'faces_boost': True, 'fast_cuts_boost': True, 'hook_critical': True, 'saturation_boost': False,
        'weights': {'duration': 0.28, 'aspect_ratio': 0.22, 'resolution': 0.10, 'audio': 0.10, 'engagement': 0.20, 'platform_bonus': 0.10},
        'tips': {
            'vertical': 'MUST be 9:16 vertical — Snapchat is full-screen only',
            'duration_short': 'Keep to 3-10 seconds — Snapchat is ultra-short form',
            'duration_long': 'Way too long for Snapchat — trim to under 10 seconds',
            'audio': 'Audio adds energy — use music or voice',
            'hook': 'Instant impact — viewers swipe in under 2 seconds',
            'cuts': 'Quick, energetic cuts match Snapchat energy',
            'faces': 'Face content thrives on Snapchat',
            'captions': 'Keep text minimal and large for mobile',
            'brightness': 'Bright, fun, and energetic lighting',
            'motion': 'High energy movement is expected',
            'resolution': 'Full HD vertical fills the screen',
            'saturation': 'Fun, vibrant colors match Snapchat style',
        },
    },
}


# ==================== ALGORITHM: Video Analyzer ====================
class VideoAnalyzer:
    """Analyze video properties for social media algorithm compatibility."""

    def __init__(self, video_path):
        self.video_path = video_path
        self.props = {}

    def _update_progress(self, scan_id, phase, detail, pct):
        if not scan_id:
            return
        with _algorithm_lock:
            if scan_id in _algorithm_progress:
                _algorithm_progress[scan_id].update({
                    'phase': phase, 'phase_detail': detail, 'percent': int(pct)
                })

    def analyze(self, scan_id=None):
        """Run full analysis pipeline."""
        self._update_progress(scan_id, 'analyzing', 'Reading video metadata...', 5)
        self._extract_basic_props()

        self._update_progress(scan_id, 'analyzing', 'Analyzing audio track...', 15)
        self._analyze_audio()

        self._update_progress(scan_id, 'analyzing', 'Analyzing visual properties...', 25)
        self._analyze_visuals(scan_id)

        self._update_progress(scan_id, 'analyzing', 'Detecting faces...', 65)
        self._detect_faces()

        self._update_progress(scan_id, 'analyzing', 'Checking for text overlays...', 80)
        self._detect_text_overlays()

        self._update_progress(scan_id, 'analyzing', 'Analyzing opening hook...', 88)
        self._analyze_hook()

        self._update_progress(scan_id, 'scoring', 'Calculating platform scores...', 92)
        scores = self._score_all_platforms()

        self._update_progress(scan_id, 'done', 'Analysis complete', 100)
        return {'properties': self.props, 'scores': scores}

    def _extract_basic_props(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError("Cannot open video file")
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()

        file_size = os.path.getsize(self.video_path)
        bitrate = (file_size * 8 / duration / 1000) if duration > 0 else 0

        aspect = w / h if h > 0 else 1.0
        if aspect < 0.7:
            orientation = 'vertical'
        elif aspect > 1.3:
            orientation = 'horizontal'
        else:
            orientation = 'square'

        self.props.update({
            'width': w, 'height': h, 'fps': round(fps, 1),
            'duration': round(duration, 1), 'frame_count': frame_count,
            'bitrate_kbps': round(bitrate), 'file_size_mb': round(file_size / (1024*1024), 1),
            'aspect_ratio': round(aspect, 2), 'orientation': orientation,
        })

    def _analyze_audio(self):
        try:
            import av
            container = av.open(self.video_path)
            audio_streams = [s for s in container.streams if s.type == 'audio']
            if not audio_streams:
                container.close()
                self.props['has_audio'] = False
                self.props['audio_loudness'] = 0
                return
            self.props['has_audio'] = True
            # Compute RMS loudness from first 30s
            resampler = av.AudioResampler(format='s16', layout='mono', rate=16000)
            total_sq = 0.0
            sample_count = 0
            max_samples = 16000 * 30  # 30 seconds
            for frame in container.decode(audio=0):
                resampled = resampler.resample(frame)
                for rf in (resampled if isinstance(resampled, list) else [resampled]):
                    arr = rf.to_ndarray().flatten().astype(float)
                    total_sq += (arr ** 2).sum()
                    sample_count += len(arr)
                    if sample_count >= max_samples:
                        break
                if sample_count >= max_samples:
                    break
            container.close()
            rms = (total_sq / max(sample_count, 1)) ** 0.5
            loudness = min(100, rms / 3276.8 * 100)
            self.props['audio_loudness'] = round(loudness, 1)
        except Exception:
            self.props['has_audio'] = False
            self.props['audio_loudness'] = 0

    def _analyze_visuals(self, scan_id=None):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        # Sample 1 frame per second, max 60
        sample_interval = max(1.0, duration / 60) if duration > 60 else 1.0
        n_samples = min(60, int(duration / sample_interval)) if duration > 0 else 0

        brightness_vals, contrast_vals, saturation_vals = [], [], []
        motion_vals = []
        scene_changes = 0
        prev_gray = None
        SCENE_THRESH = 30

        for i in range(n_samples):
            t = i * sample_interval
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, frame = cap.read()
            if not ret:
                break

            # Update progress 25% -> 65%
            if scan_id and n_samples > 0:
                pct = 25 + int(40 * i / n_samples)
                self._update_progress(scan_id, 'analyzing', f'Analyzing frame {i+1}/{n_samples}...', pct)

            # Brightness + saturation from HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            brightness_vals.append(hsv[:,:,2].mean())
            saturation_vals.append(hsv[:,:,1].mean())

            # Contrast from grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            contrast_vals.append(gray.std())

            # Motion + scene changes via frame diff (small resize for speed)
            small = cv2.resize(gray, (160, 90))
            if prev_gray is not None:
                diff = cv2.absdiff(small, prev_gray).mean()
                motion_vals.append(diff)
                if diff > SCENE_THRESH:
                    scene_changes += 1
            prev_gray = small

        cap.release()

        self.props['avg_brightness'] = round(sum(brightness_vals) / max(len(brightness_vals), 1), 1)
        self.props['avg_contrast'] = round(sum(contrast_vals) / max(len(contrast_vals), 1), 1)
        self.props['avg_saturation'] = round(sum(saturation_vals) / max(len(saturation_vals), 1), 1)
        self.props['avg_motion'] = round(sum(motion_vals) / max(len(motion_vals), 1), 1)
        self.props['scene_changes'] = scene_changes
        self.props['scene_change_rate'] = round(scene_changes / max(duration, 1) * 60, 1)  # per minute

    def _detect_faces(self):
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(cascade_path)
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.props['face_detected'] = False
                self.props['face_frequency'] = 0
                return
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0

            interval = 3.0  # check every 3 seconds
            n_checks = min(20, int(duration / interval)) if duration > 0 else 0
            faces_found = 0

            for i in range(max(1, n_checks)):
                t = i * interval
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                ret, frame = cap.read()
                if not ret:
                    break
                # Resize for speed
                h, w = frame.shape[:2]
                scale = 320 / max(w, 1)
                small = cv2.resize(frame, (int(w * scale), int(h * scale)))
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))
                if len(faces) > 0:
                    faces_found += 1
            cap.release()
            total_checks = max(1, n_checks)
            self.props['face_detected'] = faces_found > 0
            self.props['face_frequency'] = round(faces_found / total_checks, 2)
        except Exception:
            self.props['face_detected'] = False
            self.props['face_frequency'] = 0

    def _detect_text_overlays(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.props['has_text_overlay'] = False
                self.props['text_overlay_frequency'] = 0
                return
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0

            interval = 5.0
            n_checks = min(10, int(duration / interval)) if duration > 0 else 0
            text_found = 0

            for i in range(max(1, n_checks)):
                t = i * interval
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                ret, frame = cap.read()
                if not ret:
                    break
                h, w = frame.shape[:2]
                # Bottom 25% is caption zone
                caption_zone = frame[int(h * 0.75):, :]
                gray = cv2.cvtColor(caption_zone, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                edge_density = edges.mean() / 255.0
                if edge_density > 0.08:
                    text_found += 1
            cap.release()
            total_checks = max(1, n_checks)
            self.props['has_text_overlay'] = text_found > 0
            self.props['text_overlay_frequency'] = round(text_found / total_checks, 2)
        except Exception:
            self.props['has_text_overlay'] = False
            self.props['text_overlay_frequency'] = 0

    def _analyze_hook(self):
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.props['hook_strength'] = 0
                return
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            # Read 4 frames from first 3 seconds
            frames = []
            for i in range(4):
                cap.set(cv2.CAP_PROP_POS_MSEC, i * 750)
                ret, frame = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(cv2.resize(gray, (160, 90)))
            cap.release()
            if len(frames) < 2:
                self.props['hook_strength'] = 0
                return
            diffs = []
            for i in range(1, len(frames)):
                diffs.append(cv2.absdiff(frames[i], frames[i-1]).mean())
            avg_diff = sum(diffs) / len(diffs)
            self.props['hook_strength'] = min(100, round(avg_diff * 5))
        except Exception:
            self.props['hook_strength'] = 0

    def _score_all_platforms(self):
        results = {}
        for key, spec in PLATFORM_SPECS.items():
            results[key] = self._score_platform(spec)
        return results

    def _score_platform(self, spec):
        p = self.props
        weights = spec['weights']
        breakdown = {}
        tips = []

        # --- Duration Score ---
        dur = p.get('duration', 0)
        ideal_lo, ideal_hi = spec['ideal_duration']
        ok_lo, ok_hi = spec['ok_duration']
        if ideal_lo <= dur <= ideal_hi:
            breakdown['duration'] = 100
        elif ok_lo <= dur <= ok_hi:
            if dur < ideal_lo:
                breakdown['duration'] = max(40, 100 - int((ideal_lo - dur) / max(ideal_lo - ok_lo, 1) * 60))
                tips.append(spec['tips']['duration_short'])
            else:
                breakdown['duration'] = max(40, 100 - int((dur - ideal_hi) / max(ok_hi - ideal_hi, 1) * 60))
                tips.append(spec['tips']['duration_long'])
        else:
            breakdown['duration'] = 20
            if dur < ok_lo:
                tips.append(spec['tips']['duration_short'])
            else:
                tips.append(spec['tips']['duration_long'])

        # --- Aspect Ratio Score ---
        actual_ar = p.get('aspect_ratio', 1.0)
        best_match = 999
        ideal_ar = spec['ideal_aspect'][0] / spec['ideal_aspect'][1]
        best_match = abs(actual_ar - ideal_ar) / max(ideal_ar, 0.01)
        for alt in spec.get('alt_aspects', []):
            alt_ar = alt[0] / alt[1]
            diff = abs(actual_ar - alt_ar) / max(alt_ar, 0.01)
            if diff < best_match:
                best_match = diff
        tol = spec['aspect_tolerance']
        if best_match < tol * 0.3:
            breakdown['aspect_ratio'] = 100
        elif best_match < tol:
            breakdown['aspect_ratio'] = max(50, 100 - int((best_match / tol) * 50))
        elif best_match < tol * 2:
            breakdown['aspect_ratio'] = max(25, 50 - int(((best_match - tol) / tol) * 25))
        else:
            breakdown['aspect_ratio'] = 15
        if breakdown['aspect_ratio'] < 70:
            tips.append(spec['tips']['vertical'])

        # --- Resolution Score ---
        w, h = p.get('width', 0), p.get('height', 0)
        ideal_w, ideal_h = spec['ideal_resolution']
        min_w, min_h = spec['min_resolution']
        pixels = w * h
        ideal_pixels = ideal_w * ideal_h
        min_pixels = min_w * min_h
        if pixels >= ideal_pixels:
            breakdown['resolution'] = 100
        elif pixels >= min_pixels:
            breakdown['resolution'] = 50 + int(50 * (pixels - min_pixels) / max(ideal_pixels - min_pixels, 1))
        else:
            breakdown['resolution'] = max(10, int(50 * pixels / max(min_pixels, 1)))
            tips.append(spec['tips']['resolution'])

        # --- Audio Score ---
        has_audio = p.get('has_audio', False)
        loudness = p.get('audio_loudness', 0)
        if spec['audio_required']:
            if not has_audio:
                breakdown['audio'] = 10
                tips.append(spec['tips']['audio'])
            elif loudness < 10:
                breakdown['audio'] = 35
                tips.append(spec['tips']['audio'])
            elif loudness < 25:
                breakdown['audio'] = 60
            else:
                breakdown['audio'] = min(100, 70 + int(loudness * 0.3))
        else:
            if has_audio:
                breakdown['audio'] = 80
            else:
                breakdown['audio'] = 50

        # --- Engagement Score ---
        motion_raw = p.get('avg_motion', 0)
        motion_score = min(100, int(motion_raw * 4))
        if motion_score < 40:
            tips.append(spec['tips']['motion'])

        cut_rate = p.get('scene_change_rate', 0)
        if spec['fast_cuts_boost']:
            cut_score = min(100, int(cut_rate * 4))
            if cut_score < 40:
                tips.append(spec['tips']['cuts'])
        else:
            if cut_rate < 5:
                cut_score = 50
            elif cut_rate < 20:
                cut_score = 80
            else:
                cut_score = max(40, 80 - int((cut_rate - 20) * 2))

        face_freq = p.get('face_frequency', 0)
        if spec['faces_boost']:
            face_score = min(100, int(face_freq * 150))
            if face_score < 40:
                tips.append(spec['tips']['faces'])
        else:
            face_score = 60 + int(face_freq * 40)

        hook = p.get('hook_strength', 0)
        if spec['hook_critical']:
            hook_score = hook
            if hook_score < 40:
                tips.append(spec['tips']['hook'])
        else:
            hook_score = max(60, hook)

        breakdown['engagement'] = int((motion_score + cut_score + face_score + hook_score) / 4)

        # --- Platform Bonus Score ---
        bright = p.get('avg_brightness', 128)
        if 100 <= bright <= 180:
            bright_score = 100
        elif 70 <= bright <= 200:
            bright_score = 70
        else:
            bright_score = 30
            tips.append(spec['tips']['brightness'])

        sat = p.get('avg_saturation', 80)
        if spec['saturation_boost']:
            sat_score = min(100, int(sat * 0.8))
            if sat_score < 50:
                tips.append(spec['tips']['saturation'])
        else:
            sat_score = min(100, 50 + int(sat * 0.4))

        has_captions = p.get('has_text_overlay', False)
        caption_freq = p.get('text_overlay_frequency', 0)
        if spec['captions_critical']:
            if has_captions and caption_freq > 0.3:
                caption_score = 100
            elif has_captions:
                caption_score = 60
            else:
                caption_score = 15
                tips.append(spec['tips']['captions'])
        elif spec['captions_boost']:
            caption_score = 80 if has_captions else 50
            if not has_captions:
                tips.append(spec['tips']['captions'])
        else:
            caption_score = 70

        vid_fps = p.get('fps', 30)
        fps_lo, fps_hi = spec['ideal_fps']
        if fps_lo <= vid_fps <= fps_hi:
            fps_score = 100
        elif vid_fps >= 15:
            fps_score = 70
        else:
            fps_score = 30

        breakdown['platform_bonus'] = int((bright_score + sat_score + caption_score + fps_score) / 4)

        # --- Weighted Total ---
        total = 0
        for factor, weight in weights.items():
            total += breakdown.get(factor, 50) * weight
        total = int(min(100, max(0, total)))

        # Grade
        if total >= 85:
            grade = 'A'
        elif total >= 70:
            grade = 'B'
        elif total >= 50:
            grade = 'C'
        elif total >= 30:
            grade = 'D'
        else:
            grade = 'F'

        # Deduplicate and limit tips
        seen = set()
        unique_tips = []
        for t in tips:
            if t not in seen:
                seen.add(t)
                unique_tips.append(t)
        unique_tips = unique_tips[:5]

        return {
            'name': spec['name'], 'icon': spec['icon'],
            'score': total, 'grade': grade,
            'breakdown': breakdown, 'tips': unique_tips,
        }


HTML_PAGE = r'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SEARCH SCANNER</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Inter:wght@300;400;500&display=swap');

  /* ========== MATRIX THEME (default) ========== */
  :root {
    --bg: #000a00;
    --bg-panel: rgba(0, 15, 0, 0.7);
    --bg-header: rgba(0, 10, 0, 0.85);
    --bg-input: rgba(0, 5, 0, 0.9);
    --bg-overlay: rgba(0,0,0,0.85);
    --accent: #00ff41;
    --accent-secondary: #7fff00;
    --accent-dim: #0a6e2a;
    --accent-dimmer: #0a3e1a;
    --accent-faint: #0a4e1a;
    --accent-glow: #008f11;
    --text: #00ff41;
    --text-dim: #0a8e3a;
    --text-dimmer: #0a5e2a;
    --text-faintest: #0a4e2a;
    --border: #0a3e1a;
    --border-accent: #0a6e2a;
    --hover-bg: rgba(0, 30, 0, 0.8);
    --hover-glow: rgba(0, 255, 65, 0.15);
    --error: #ff6e41;
    --glow-shadow: rgba(0, 255, 65, 0.15);
    --glow-strong: rgba(0, 255, 65, 0.3);
    --glow-text: 0 0 10px #00ff41, 0 0 30px #008f11;
    --mark-bg: rgba(0,255,65,0.25);
    --scanline-bg: repeating-linear-gradient(0deg, rgba(0,0,0,0.15) 0px, rgba(0,0,0,0.15) 1px, transparent 1px, transparent 3px);
    --badge-audio: #41a0ff;
    --badge-video: #ff6e41;
    --fill-gradient: linear-gradient(90deg, #00ff41, #7fff00);
    --fill-glow: 0 0 12px #00ff41, 0 0 4px #00ff41 inset;
    --font: 'Share Tech Mono', 'Courier New', monospace;
    --panel-blur: blur(4px);
    --header-blur: blur(6px);
    --panel-shadow: none;
    --panel-border-top: none;
    --modal-bg: #000a00;
    --scroll-track: #000a00;
    --scroll-thumb: #0a3e1a;
    --hist-name-bg: rgba(0, 5, 0, 0.8);
    --canvas-opacity: 0.12;
  }

  /* ========== GLASS THEME ========== */
  body.glass {
    --bg: #e8ecf0;
    --bg-panel: rgba(255, 255, 255, 0.45);
    --bg-header: rgba(255, 255, 255, 0.55);
    --bg-input: rgba(255, 255, 255, 0.5);
    --bg-overlay: rgba(240,242,245,0.92);
    --accent: #2a6cb6;
    --accent-secondary: #3d8fe0;
    --accent-dim: rgba(42, 108, 182, 0.55);
    --accent-dimmer: rgba(42, 108, 182, 0.2);
    --accent-faint: rgba(42, 108, 182, 0.08);
    --accent-glow: #3d8fe0;
    --text: #1a2a3a;
    --text-dim: rgba(30, 50, 70, 0.7);
    --text-dimmer: rgba(60, 80, 110, 0.5);
    --text-faintest: rgba(80, 100, 130, 0.35);
    --border: rgba(0, 0, 0, 0.1);
    --border-accent: rgba(42, 108, 182, 0.3);
    --hover-bg: rgba(255, 255, 255, 0.6);
    --hover-glow: rgba(42, 108, 182, 0.1);
    --error: #d44;
    --glow-shadow: rgba(42, 108, 182, 0.08);
    --glow-strong: rgba(42, 108, 182, 0.15);
    --glow-text: 0 0 8px rgba(42, 108, 182, 0.25);
    --mark-bg: rgba(42, 108, 182, 0.15);
    --scanline-bg: none;
    --badge-audio: #2a6cb6;
    --badge-video: #c44;
    --fill-gradient: linear-gradient(90deg, #2a6cb6, #3d8fe0);
    --fill-glow: 0 0 10px rgba(42, 108, 182, 0.25);
    --font: 'Inter', 'Share Tech Mono', sans-serif;
    --panel-blur: blur(24px);
    --header-blur: blur(30px);
    --panel-shadow: 0 2px 20px rgba(0,0,0,0.06), 0 1px 3px rgba(0,0,0,0.08);
    --panel-border-top: 1px solid rgba(255,255,255,0.7);
    --modal-bg: rgba(240, 242, 245, 0.95);
    --scroll-track: rgba(220, 225, 230, 0.5);
    --scroll-thumb: rgba(42, 108, 182, 0.25);
    --hist-name-bg: rgba(255, 255, 255, 0.7);
    --canvas-opacity: 0;
  }
  body.glass .app-wrap::before { display: none; }
  body.glass .header h1 { animation: none; font-family: 'Inter', sans-serif; font-weight: 300; letter-spacing: 6px;
    color: #1a2a3a; text-shadow: none; }
  body.glass .header p { font-family: 'Inter', sans-serif; }
  body.glass .upload-zone, body.glass .settings, body.glass .result-card,
  body.glass .transcript-segment, body.glass .scan-history-item,
  body.glass .history-grid, body.glass .full-transcript {
    backdrop-filter: blur(24px); -webkit-backdrop-filter: blur(24px);
    background: rgba(255, 255, 255, 0.45);
    box-shadow: var(--panel-shadow);
    border-top: var(--panel-border-top);
    border: 1px solid rgba(255,255,255,0.6);
  }
  body.glass .folder-modal { backdrop-filter: blur(30px); -webkit-backdrop-filter: blur(30px);
    background: rgba(255,255,255,0.6);
    box-shadow: 0 8px 40px rgba(0,0,0,0.1), var(--panel-shadow); }
  body.glass .logo-svg { filter: drop-shadow(0 0 6px rgba(42,108,182,0.35)); }
  body.glass .logo-svg:hover { filter: drop-shadow(0 0 10px rgba(42,108,182,0.45)) drop-shadow(0 0 20px rgba(42,108,182,0.15)); }
  body.glass .upload-zone .upload-icon svg { filter: drop-shadow(0 0 4px rgba(42,108,182,0.3)); }
  body.glass .result-info .similarity.high { color: #2a6cb6; text-shadow: none; }
  body.glass .result-info .similarity.med { color: #3d8fe0; text-shadow: none; }
  body.glass .result-info .similarity.low { color: #c44; text-shadow: none; }
  body.glass .video-badge { background: var(--badge-video); }
  body.glass .audio-badge { background: var(--badge-audio); }
  body.glass .empty-state .icon { filter: drop-shadow(0 0 4px rgba(42,108,182,0.2)); }
  body.glass .mode-tab.active { text-shadow: none; color: #2a6cb6; }
  body.glass mark { background: rgba(42,108,182,0.12); color: #2a6cb6; }
  body.glass .scan-btn { color: #2a6cb6; border-color: rgba(42,108,182,0.4); }
  body.glass .scan-btn:hover { background: #2a6cb6; color: #fff; }
  body.glass .mode-tab { color: rgba(30,50,70,0.5); }
  body.glass .mode-tab:hover { color: rgba(30,50,70,0.8); }
  body.glass .mode-tab.active { color: #2a6cb6; border-color: #2a6cb6; }
  /* Glass background - soft white gradient */
  body.glass::after { content:''; position:fixed; top:0; left:0; width:100%; height:100%;
    background: radial-gradient(ellipse at 20% 20%, rgba(42,108,182,0.06) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 80%, rgba(100,60,200,0.03) 0%, transparent 50%),
                radial-gradient(ellipse at 50% 50%, rgba(255,255,255,0.4) 0%, transparent 70%);
    pointer-events:none; z-index:0; }

  /* ========== BASE STYLES (using variables) ========== */
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: var(--font); background: var(--bg); color: var(--text);
         min-height: 100vh; position: relative; overflow-x: hidden; }
  #matrixCanvas { position: fixed; top: 0; left: 0; width: 100%; height: 100%;
                  z-index: 0; pointer-events: none; opacity: var(--canvas-opacity); transition: opacity 0.5s; }
  .app-wrap { position: relative; z-index: 1; }
  .app-wrap::before { content: ''; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    background: var(--scanline-bg); pointer-events: none; z-index: 9999; }
  .header { background: var(--bg-header); padding: 18px 30px;
            border-bottom: 2px solid var(--accent); display: flex; align-items: center; gap: 18px;
            justify-content: center; text-align: center; position: relative;
            backdrop-filter: var(--header-blur); -webkit-backdrop-filter: var(--header-blur); }
  .logo-svg { width: 56px; height: 56px; flex-shrink: 0; filter: drop-shadow(0 0 8px var(--accent)); transition: filter 0.3s; }
  .logo-svg:hover { filter: drop-shadow(0 0 14px var(--accent)) drop-shadow(0 0 30px var(--glow-shadow)); }
  .header h1 { font-size: 22px; color: var(--text); text-shadow: var(--glow-text);
               letter-spacing: 2px; text-transform: uppercase; }
  .header p { color: var(--accent-dim); font-size: 13px; letter-spacing: 1px; }
  .main { max-width: 1200px; margin: 0 auto; padding: 25px; }
  .controls { display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 25px; }
  .upload-zone { flex: 1; min-width: 280px; border: 2px dashed var(--border-accent);
                 border-radius: 4px; padding: 30px; text-align: center;
                 cursor: pointer; transition: all 0.3s; background: var(--bg-panel);
                 backdrop-filter: var(--panel-blur); -webkit-backdrop-filter: var(--panel-blur); }
  .upload-zone:hover, .upload-zone.dragover { border-color: var(--accent); background: var(--hover-bg);
    box-shadow: 0 0 20px var(--hover-glow), inset 0 0 20px rgba(0, 255, 65, 0.03); }
  .upload-zone img { max-width: 200px; max-height: 200px; border-radius: 4px; margin-top: 10px;
                     border: 1px solid var(--border-accent); }
  .upload-zone.has-image { cursor: default; }
  .change-btn { display: inline-block; margin-top: 8px; padding: 5px 14px; background: transparent;
                color: var(--accent-dim); border: 1px solid var(--border); border-radius: 4px; cursor: pointer;
                font-size: 11px; font-family: var(--font); text-transform: uppercase;
                letter-spacing: 1px; transition: all 0.3s; }
  .change-btn:hover { border-color: var(--accent); color: var(--accent); }
  .upload-zone .upload-icon { margin-bottom: 10px; }
  .upload-zone .upload-icon svg { width: 60px; height: 60px; filter: drop-shadow(0 0 6px var(--accent)); }
  .upload-zone p { color: var(--accent-dim); margin: 5px 0; font-size: 13px; }
  .upload-zone p strong { color: var(--accent); }
  .settings { flex: 1; min-width: 280px; background: var(--bg-panel); border-radius: 4px;
              padding: 25px; border: 1px solid var(--border-accent);
              backdrop-filter: var(--panel-blur); -webkit-backdrop-filter: var(--panel-blur); }
  .settings label { display: block; margin-bottom: 6px; font-weight: 600; color: var(--text);
                    text-transform: uppercase; font-size: 12px; letter-spacing: 1px; }
  .settings input[type="text"] { width: 100%; padding: 10px 12px; border-radius: 4px;
                                  border: 1px solid var(--border-accent); background: var(--bg-input);
                                  color: var(--text); font-size: 14px; margin-bottom: 15px;
                                  font-family: var(--font); }
  .settings input[type="text"]:focus { outline: none; border-color: var(--accent);
    box-shadow: 0 0 10px var(--glow-strong); }
  .settings input[type="text"]::placeholder { color: var(--accent-faint); }
  .folder-row { display: flex; gap: 8px; margin-bottom: 15px; }
  .folder-row input { flex: 1; }
  .browse-btn { padding: 10px 18px; background: transparent; color: var(--accent); border: 1px solid var(--accent);
                border-radius: 4px; cursor: pointer; font-size: 13px; white-space: nowrap;
                font-family: var(--font); text-transform: uppercase;
                letter-spacing: 1px; transition: all 0.3s; }
  .browse-btn:hover { background: var(--accent); color: var(--bg);
    box-shadow: 0 0 12px var(--glow-strong); }
  .scan-btn { width: 100%; padding: 14px; background: transparent; color: var(--accent);
              border: 2px solid var(--accent); border-radius: 4px; font-size: 15px; font-weight: 600;
              cursor: pointer; transition: all 0.3s; text-transform: uppercase; letter-spacing: 3px;
              font-family: var(--font); }
  .scan-btn:hover { background: var(--accent); color: var(--bg);
    box-shadow: 0 0 20px var(--accent), 0 0 40px var(--glow-strong); }
  .scan-btn:disabled { border-color: var(--border); color: var(--border); background: transparent;
                       cursor: not-allowed; box-shadow: none; }
  .status { text-align: center; padding: 15px; color: var(--accent-dim); font-size: 14px; }
  .status .spinner { display: inline-block; width: 20px; height: 20px;
                     border: 3px solid var(--border); border-top-color: var(--accent);
                     border-radius: 50%; animation: spin 0.8s linear infinite;
                     vertical-align: middle; margin-right: 8px; }
  @keyframes spin { to { transform: rotate(360deg); } }
  .progress-bar { width: 100%; height: 10px; background: var(--accent-dimmer); border-radius: 5px;
                  margin-top: 10px; overflow: hidden; border: 1px solid var(--accent-dim); }
  .progress-bar .fill { height: 100%; background: var(--fill-gradient);
                        transition: width 0.3s ease; border-radius: 5px;
                        box-shadow: var(--fill-glow); }
  .results { display: flex; flex-direction: column; gap: 12px; }
  .results-header { font-size: 16px; font-weight: 600; margin-bottom: 5px; color: var(--text);
                    text-transform: uppercase; letter-spacing: 1px; text-shadow: 0 0 6px var(--accent); }
  .result-card { display: flex; gap: 16px; background: var(--bg-panel); border-radius: 4px;
                 padding: 14px; border: 1px solid var(--border); transition: all 0.3s;
                 align-items: center; backdrop-filter: var(--panel-blur); -webkit-backdrop-filter: var(--panel-blur); }
  .result-card:hover { border-color: var(--accent); box-shadow: 0 0 15px var(--hover-glow); }
  .result-card img { width: 120px; height: 120px; object-fit: cover; border-radius: 4px;
                     flex-shrink: 0; background: #000; border: 1px solid var(--border); }
  .result-info { flex: 1; min-width: 0; }
  .result-info .similarity { font-size: 15px; font-weight: 700; margin-bottom: 4px; }
  .result-info .similarity.high { color: #00ff41; text-shadow: 0 0 8px #00ff41; }
  .result-info .similarity.med { color: #7fff00; text-shadow: 0 0 6px #7fff00; }
  .result-info .similarity.low { color: var(--error); text-shadow: 0 0 6px var(--error); }
  .result-info .filepath { color: var(--text-dim); font-size: 12px; word-break: break-all; margin-bottom: 4px; }
  .result-info .folder { color: var(--text-dimmer); font-size: 11px; word-break: break-all; }
  .reveal-btn { padding: 8px 16px; background: transparent; color: var(--accent);
                border: 1px solid var(--accent); border-radius: 4px; cursor: pointer;
                font-size: 12px; white-space: nowrap; flex-shrink: 0;
                font-family: var(--font); text-transform: uppercase;
                letter-spacing: 1px; transition: all 0.3s; }
  .reveal-btn:hover { background: var(--accent); color: var(--bg);
    box-shadow: 0 0 12px var(--glow-strong); }
  .empty-state { text-align: center; padding: 60px 20px; color: var(--text-faintest); }
  .empty-state .icon { font-size: 64px; margin-bottom: 15px; filter: drop-shadow(0 0 6px var(--accent-dim)); }
  .empty-state p { font-size: 14px; }
  .history-section { margin-bottom: 25px; }
  .history-toggle { background: transparent; color: var(--accent-dim); border: 1px solid var(--border);
                    border-radius: 4px; padding: 8px 16px; cursor: pointer; font-size: 13px;
                    font-family: var(--font); text-transform: uppercase;
                    letter-spacing: 1px; transition: all 0.3s; width: 100%; text-align: left; }
  .history-toggle:hover { border-color: var(--accent); color: var(--accent); }
  .history-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(100px, 1fr));
                  gap: 10px; margin-top: 12px; padding: 12px; background: var(--bg-panel);
                  border: 1px solid var(--border); border-radius: 4px; }
  .history-item { cursor: pointer; border: 2px solid transparent; border-radius: 4px;
                  overflow: hidden; transition: all 0.3s; position: relative; }
  .history-item:hover { border-color: var(--accent); box-shadow: 0 0 10px var(--glow-strong); }
  .history-item img { width: 100%; height: 80px; object-fit: cover; display: block; }
  .history-item .hist-name { font-size: 9px; color: var(--accent-dim); padding: 3px 4px;
                             white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
                             background: var(--hist-name-bg); }
  .history-item .hist-delete { position: absolute; top: 3px; right: 3px; width: 18px; height: 18px;
    background: rgba(0,0,0,0.7); border: 1px solid var(--error); border-radius: 50%; color: var(--error);
    font-size: 11px; line-height: 16px; text-align: center; cursor: pointer; display: none;
    font-family: var(--font); transition: all 0.2s; z-index: 2; }
  .history-item .hist-delete:hover { background: var(--error); color: #000; }
  .history-item:hover .hist-delete { display: block; }
  .history-path { font-size: 11px; color: var(--text-dimmer); margin-top: 6px; }
  .history-controls { display: flex; gap: 10px; align-items: center; margin-top: 8px; margin-bottom: 4px; }
  .history-clear-btn { background: transparent; color: var(--error); border: 1px solid var(--error);
    border-radius: 4px; padding: 5px 12px; cursor: pointer; font-size: 11px;
    font-family: var(--font); text-transform: uppercase;
    letter-spacing: 1px; transition: all 0.3s; }
  .history-clear-btn:hover { background: var(--error); color: #000; }
  .scan-history-item { padding: 10px 12px; border: 1px solid var(--border); border-radius: 4px;
    margin-bottom: 8px; background: var(--bg-panel); transition: border-color 0.3s; }
  .scan-history-item:hover { border-color: var(--accent); }
  .video-badge { display: inline-block; background: var(--badge-video); color: #000; padding: 2px 8px;
    border-radius: 3px; font-size: 10px; font-weight: 700; letter-spacing: 1px; margin-right: 8px;
    text-shadow: none; vertical-align: middle; }
  .timestamp { color: var(--accent-secondary); font-size: 11px; margin-top: 4px; }

  @keyframes flicker { 0%, 95% { opacity: 1; } 96% { opacity: 0.8; } 97% { opacity: 1; }
    98% { opacity: 0.6; } 100% { opacity: 1; } }
  .header h1 { animation: flicker 4s infinite; }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

  /* Folder browser modal */
  .modal-overlay { display: none; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    background: var(--bg-overlay); z-index: 10000; justify-content: center; align-items: center; }
  .modal-overlay.open { display: flex; }
  .folder-modal { background: var(--modal-bg); border: 2px solid var(--accent); border-radius: 8px;
    width: 550px; max-width: 90vw; max-height: 80vh; display: flex; flex-direction: column;
    box-shadow: 0 0 40px var(--glow-shadow); }
  .folder-modal-header { padding: 14px 18px; border-bottom: 1px solid var(--border);
    display: flex; justify-content: space-between; align-items: center; }
  .folder-modal-header h2 { font-size: 15px; color: var(--accent); letter-spacing: 1px; }
  .modal-close { background: transparent; border: 1px solid var(--border); color: var(--accent-dim);
    padding: 4px 12px; cursor: pointer; font-family: var(--font);
    font-size: 12px; border-radius: 4px; transition: all 0.3s; }
  .modal-close:hover { border-color: var(--error); color: var(--error); }
  .folder-modal-path { padding: 10px 18px; border-bottom: 1px solid var(--border);
    font-size: 12px; color: var(--text-dim); word-break: break-all; background: var(--bg-panel); }
  .folder-modal-list { flex: 1; overflow-y: auto; padding: 8px 0; min-height: 200px; max-height: 50vh; }
  .folder-modal-list::-webkit-scrollbar { width: 8px; }
  .folder-modal-list::-webkit-scrollbar-track { background: var(--scroll-track); }
  .folder-modal-list::-webkit-scrollbar-thumb { background: var(--scroll-thumb); border-radius: 4px; }
  .folder-item { padding: 8px 18px; cursor: pointer; display: flex; align-items: center;
    gap: 10px; font-size: 13px; color: var(--accent); transition: background 0.15s; }
  .folder-item:hover { background: var(--hover-bg); }
  .folder-item .icon { font-size: 16px; flex-shrink: 0; width: 20px; text-align: center; }
  .folder-item.parent { color: var(--text-dim); }
  .folder-modal-footer { padding: 12px 18px; border-top: 1px solid var(--border);
    display: flex; justify-content: flex-end; gap: 10px; }
  .modal-select-btn { padding: 10px 24px; background: transparent; color: var(--accent);
    border: 2px solid var(--accent); border-radius: 4px; cursor: pointer; font-size: 13px;
    font-family: var(--font); text-transform: uppercase;
    letter-spacing: 1px; transition: all 0.3s; }
  .modal-select-btn:hover { background: var(--accent); color: var(--bg);
    box-shadow: 0 0 12px var(--glow-strong); }

  /* Mode tabs */
  .mode-tabs { display:flex; gap:0; margin-bottom:25px; border-bottom:2px solid var(--border); }
  .mode-tab { flex:1; padding:12px 20px; background:transparent; color:var(--accent-dim);
    border:none; border-bottom:2px solid transparent; cursor:pointer;
    font-family:var(--font); font-size:14px;
    text-transform:uppercase; letter-spacing:2px; transition:all 0.3s; margin-bottom:-2px; }
  .mode-tab:hover { color:var(--accent); }
  .mode-tab.active { color:var(--accent); border-bottom-color:var(--accent); text-shadow:0 0 8px var(--accent); }
  .mode-panel { display:none; }
  .mode-panel.active { display:block; }

  /* Audio scan styles */
  .search-text-input { width:100%; padding:10px 12px; border-radius:4px;
    border:1px solid var(--border-accent); background:var(--bg-input);
    color:var(--text); font-size:14px; margin-bottom:15px;
    font-family:var(--font); }
  .search-text-input:focus { outline:none; border-color:var(--accent);
    box-shadow:0 0 10px var(--glow-strong); }
  .search-text-input::placeholder { color:var(--accent-faint); }
  .audio-badge { display:inline-block; background:var(--badge-audio); color:#000;
    padding:2px 8px; border-radius:3px; font-size:10px; font-weight:700;
    letter-spacing:1px; margin-right:8px; }
  .transcript-segment { background:var(--bg-panel); border:1px solid var(--border);
    border-radius:4px; padding:14px; margin-bottom:8px; transition:all 0.3s; }
  .transcript-segment:hover { border-color:var(--accent); box-shadow:0 0 15px var(--hover-glow); }
  .transcript-segment mark { background:var(--mark-bg); color:var(--accent);
    padding:1px 3px; border-radius:2px; font-weight:700; }
  .transcript-segment .seg-time { color:var(--accent-secondary); font-size:12px; font-weight:600; margin-bottom:6px; }
  .transcript-segment .seg-text { color:var(--text-dim); font-size:14px; line-height:1.6; }
  .transcript-segment .matched-terms { font-size:11px; color:var(--text-dimmer); margin-top:6px; }
  .full-transcript { margin-top:10px; border:1px solid var(--border); border-radius:4px;
    padding:16px; background:var(--bg-panel); max-height:400px;
    overflow-y:auto; font-size:13px; line-height:1.8; color:var(--accent-dim); }

  /* Theme selector dropdown */
  .theme-selector { position:relative; margin-left:16px; flex-shrink:0; }
  .theme-btn {
    background:var(--bg-panel); border:1px solid var(--border); color:var(--accent-dim);
    padding:6px 14px; border-radius:20px; cursor:pointer; font-family:var(--font);
    font-size:11px; letter-spacing:1px; transition:all 0.3s; display:flex; align-items:center; gap:6px;
    backdrop-filter:blur(8px); -webkit-backdrop-filter:blur(8px); }
  .theme-btn:hover { border-color:var(--accent); color:var(--accent);
    box-shadow:0 0 12px var(--glow-shadow); }
  .theme-btn .theme-dot { width:8px; height:8px; border-radius:50%;
    background:var(--accent); box-shadow:0 0 6px var(--accent); transition:all 0.3s; }
  .theme-btn .arrow { font-size:8px; margin-left:2px; transition:transform 0.2s; }
  .theme-selector.open .theme-btn .arrow { transform:rotate(180deg); }
  .theme-menu { position:absolute; top:calc(100% + 8px); right:0; min-width:160px;
    background:var(--bg-panel); border:1px solid var(--border); border-radius:8px;
    backdrop-filter:blur(20px); -webkit-backdrop-filter:blur(20px);
    box-shadow:0 8px 32px rgba(0,0,0,0.4); padding:6px 0;
    opacity:0; visibility:hidden; transform:translateY(-6px);
    transition:all 0.2s ease; z-index:100; }
  .theme-selector.open .theme-menu { opacity:1; visibility:visible; transform:translateY(0); }
  .theme-option { display:flex; align-items:center; gap:10px; padding:10px 16px;
    cursor:pointer; transition:background 0.15s; font-family:var(--font);
    font-size:12px; color:var(--accent-dim); letter-spacing:1px; }
  .theme-option:hover { background:var(--hover-bg); color:var(--accent); }
  .theme-option.active { color:var(--accent); }
  .theme-option .opt-dot { width:10px; height:10px; border-radius:50%; flex-shrink:0;
    border:2px solid currentColor; transition:all 0.2s; }
  .theme-option.active .opt-dot { background:var(--accent); box-shadow:0 0 8px var(--accent); }
  .theme-option .opt-label { flex:1; }
  .theme-option .opt-check { font-size:14px; opacity:0; transition:opacity 0.2s; }
  .theme-option.active .opt-check { opacity:1; }
  body.glass .theme-btn { background:rgba(255,255,255,0.5); border-color:rgba(0,0,0,0.1); color:#2a6cb6; }
  body.glass .theme-btn:hover { border-color:#2a6cb6; box-shadow:0 0 8px rgba(42,108,182,0.15); }
  body.glass .theme-btn .theme-dot { box-shadow:0 0 4px rgba(42,108,182,0.4); }
  body.glass .theme-menu { background:rgba(255,255,255,0.75); box-shadow:0 8px 32px rgba(0,0,0,0.1); border-color:rgba(0,0,0,0.08); }
  body.glass .theme-option { color:rgba(30,50,70,0.6); }
  body.glass .theme-option:hover { background:rgba(42,108,182,0.06); color:#2a6cb6; }
  body.glass .theme-option.active { color:#2a6cb6; }
  body.glass .theme-option.active .opt-dot { box-shadow:0 0 6px rgba(42,108,182,0.4); }

  /* ==================== Algorithm Tab Styles ==================== */
  .algo-props { display:grid; grid-template-columns:repeat(auto-fill, minmax(180px,1fr)); gap:8px; margin-bottom:20px; }
  .algo-prop { background:var(--bg-panel); border:1px solid var(--border); border-radius:6px; padding:10px 14px;
    font-size:11px; display:flex; justify-content:space-between; align-items:center; }
  .algo-prop .prop-label { color:var(--text-dimmer); text-transform:uppercase; letter-spacing:1px; font-size:10px; }
  .algo-prop .prop-value { color:var(--accent); font-weight:600; font-size:13px; }
  .algo-summary { display:grid; grid-template-columns:repeat(auto-fill, minmax(160px,1fr)); gap:14px; margin-bottom:24px; }
  .algo-card { background:var(--bg-panel); border:1px solid var(--border); border-radius:10px; padding:16px 12px;
    text-align:center; cursor:pointer; transition:all 0.3s; position:relative; overflow:hidden; }
  .algo-card:hover { border-color:var(--accent); box-shadow:0 0 20px var(--glow-shadow);
    transform:translateY(-2px); }
  .algo-card .card-icon { font-size:22px; margin-bottom:6px; }
  .algo-card .card-name { font-size:11px; color:var(--text-dim); text-transform:uppercase;
    letter-spacing:1px; margin-bottom:10px; }
  .algo-card .card-ring { margin:0 auto 8px; }
  .algo-card .card-score { font-size:24px; font-weight:700; font-family:var(--font); }
  .algo-card .card-grade { font-size:13px; font-weight:700; letter-spacing:2px; margin-top:4px; }
  .grade-a { color:#00ff41; text-shadow:0 0 8px rgba(0,255,65,0.4); }
  .grade-b { color:#7fff00; text-shadow:0 0 8px rgba(127,255,0,0.3); }
  .grade-c { color:#ffaa00; text-shadow:0 0 8px rgba(255,170,0,0.3); }
  .grade-d { color:#ff6e41; text-shadow:0 0 8px rgba(255,110,65,0.3); }
  .grade-f { color:#ff4141; text-shadow:0 0 8px rgba(255,65,65,0.3); }
  body.glass .grade-a { color:#1a8a3e; text-shadow:none; }
  body.glass .grade-b { color:#5a9a10; text-shadow:none; }
  body.glass .grade-c { color:#b87a00; text-shadow:none; }
  body.glass .grade-d { color:#c44; text-shadow:none; }
  body.glass .grade-f { color:#a00; text-shadow:none; }
  .algo-details { margin-top:10px; }
  .algo-detail { background:var(--bg-panel); border:1px solid var(--border); border-radius:10px;
    padding:18px; margin-bottom:14px; transition:all 0.3s; }
  .algo-detail:hover { border-color:var(--accent-dim); }
  .algo-detail-header { display:flex; align-items:center; gap:12px; margin-bottom:14px;
    padding-bottom:12px; border-bottom:1px solid var(--border); }
  .algo-detail-header .detail-icon { font-size:24px; }
  .algo-detail-header .detail-name { font-size:14px; font-weight:600; color:var(--text);
    letter-spacing:1px; flex:1; }
  .algo-detail-header .detail-badge { font-size:18px; font-weight:700; padding:4px 14px;
    border-radius:20px; border:2px solid currentColor; }
  .algo-breakdown { display:grid; grid-template-columns:1fr 1fr; gap:8px 16px; margin-bottom:14px; }
  @media (max-width:600px) { .algo-breakdown { grid-template-columns:1fr; } }
  .algo-bar-row { display:flex; align-items:center; gap:8px; }
  .algo-bar-label { font-size:10px; color:var(--text-dimmer); text-transform:uppercase;
    letter-spacing:1px; width:80px; flex-shrink:0; }
  .algo-bar-track { flex:1; height:8px; background:var(--accent-faint); border-radius:4px; overflow:hidden; }
  .algo-bar-fill { height:100%; border-radius:4px; transition:width 0.6s ease; }
  .algo-bar-val { font-size:11px; font-weight:600; color:var(--accent); width:28px; text-align:right; flex-shrink:0; }
  .algo-tips { list-style:none; padding:0; margin:0; }
  .algo-tips li { padding:8px 12px; margin-bottom:6px; border-left:3px solid var(--accent);
    background:var(--accent-faint); border-radius:0 6px 6px 0; font-size:12px; color:var(--text-dim);
    line-height:1.5; }
  .algo-tips li::before { content:'> '; color:var(--accent); font-weight:700; }
  .algo-no-tips { font-size:12px; color:var(--accent); padding:8px 0; }
  body.glass .algo-card { background:rgba(255,255,255,0.35); border-color:rgba(255,255,255,0.5); }
  body.glass .algo-card:hover { border-color:#2a6cb6; box-shadow:0 0 15px rgba(42,108,182,0.15); }
  body.glass .algo-detail { background:rgba(255,255,255,0.35); border-color:rgba(255,255,255,0.5); }
  body.glass .algo-prop { background:rgba(255,255,255,0.35); border-color:rgba(255,255,255,0.5); }
  body.glass .algo-tips li { background:rgba(42,108,182,0.06); border-left-color:#2a6cb6; }
</style>
</head>
<body>

<canvas id="matrixCanvas"></canvas>

<div class="app-wrap">
<div class="header">
  <svg class="logo-svg" id="logoSvg" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
    <circle cx="40" cy="40" r="28" fill="none" stroke="#00ff41" stroke-width="4" class="logo-stroke"/>
    <line x1="60" y1="60" x2="88" y2="88" stroke="#00ff41" stroke-width="6" stroke-linecap="round" class="logo-stroke"/>
    <path d="M22 40 Q40 25 58 40 Q40 55 22 40 Z" fill="none" stroke="#00ff41" stroke-width="2.5" class="logo-stroke"/>
    <circle id="eyeIris" cx="40" cy="40" r="7" fill="none" stroke="#00ff41" stroke-width="2" class="logo-stroke"/>
    <circle id="eyePupil" cx="40" cy="40" r="3" fill="#00ff41" class="logo-fill"/>
    <circle id="eyeGlint" cx="37" cy="37" r="1.5" fill="#aaffaa" class="logo-glint"/>
  </svg>
  <div>
    <h1>SEARCH SCANNER</h1>
    <p>UPLOAD / SEARCH / MATCH</p>
  </div>
  <div class="theme-selector" id="themeSelector">
    <button class="theme-btn" onclick="toggleThemeMenu(event)" title="Switch Theme">
      <span class="theme-dot"></span> <span id="themeLabel">MATRIX</span> <span class="arrow">&#9660;</span>
    </button>
    <div class="theme-menu" id="themeMenu">
      <div class="theme-option active" data-theme="matrix" onclick="selectTheme('matrix')">
        <span class="opt-dot" style="color:#00ff41;"></span>
        <span class="opt-label">MATRIX</span>
        <span class="opt-check">&#10003;</span>
      </div>
      <div class="theme-option" data-theme="glass" onclick="selectTheme('glass')">
        <span class="opt-dot" style="color:#2a6cb6;"></span>
        <span class="opt-label">GLASS</span>
        <span class="opt-check">&#10003;</span>
      </div>
    </div>
  </div>
</div>

<div class="main">
  <div class="mode-tabs top-mode-tabs">
    <button class="mode-tab active" onclick="switchMode('image')" id="tabImage"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:4px"><rect x="2" y="3" width="20" height="18" rx="2"/><circle cx="8.5" cy="10.5" r="2.5"/><path d="M21 15l-5-5L5 21"/></svg>Image</button>
    <button class="mode-tab" onclick="switchMode('audio')" id="tabAudio"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:4px"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.54 8.46a5 5 0 010 7.07"/><path d="M19.07 4.93a10 10 0 010 14.14"/></svg>Audio</button>
    <button class="mode-tab" onclick="switchMode('text')" id="tabText"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:4px"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>Text</button>
    <button class="mode-tab" onclick="switchMode('transcribe')" id="tabTranscribe"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:4px"><path d="M17 3a2.828 2.828 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5L17 3z"/><path d="M15 5l4 4"/></svg>Transcribe</button>
    <button class="mode-tab" onclick="switchMode('algorithm')" id="tabAlgorithm"><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:-2px;margin-right:4px"><path d="M12 20V10"/><path d="M18 20V4"/><path d="M6 20v-4"/></svg>Algorithm</button>
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

  <div class="mode-panel" id="panelAlgorithm">
    <div class="controls">
      <div class="upload-zone" id="algorithmDropZone" onclick="if(!algorithmUploadedFile) document.getElementById('algorithmFileInput').click()">
        <input type="file" id="algorithmFileInput" hidden accept=".mp4,.mov,.avi,.mkv,.webm,.m4v,.flv,.wmv">
        <div class="upload-icon" id="algorithmUploadIcon">
          <svg width="50" height="50" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" style="opacity:0.5;filter:drop-shadow(0 0 6px var(--accent))">
            <path d="M12 20V10"/><path d="M18 20V4"/><path d="M6 20v-4"/>
          </svg>
        </div>
        <p class="upload-text" id="algorithmUploadText">> Drop video file here or click to browse _</p>
        <p class="upload-hint" id="algorithmUploadHint">MP4, MOV, AVI, MKV, WebM • 500MB max</p>
      </div>
      <div class="settings" id="algorithmControls">
        <label style="font-size:12px;color:var(--text-dim);display:block;margin-bottom:10px;line-height:1.6">
          Analyzes your video against social media algorithms.<br>
          Scores your video 0-100 for each platform with tips to improve.
        </label>
        <div id="algorithmDisabledNotice" hidden style="color:var(--error);font-size:12px;margin-bottom:10px">
          &#9888; OpenCV not available. Install opencv-python to use this tool.
        </div>
        <button class="scan-btn" id="algorithmScanBtn" onclick="startAlgorithm()" disabled>[ Analyze Video ]</button>
      </div>
    </div>
    <div class="results" id="algorithmResults">
      <div class="status-line" id="algorithmStatus" style="display:none">
        <span class="spinner" id="algorithmSpinner"></span>
        <span id="algorithmStatusText">Analyzing...</span>
        <span id="algorithmTimerDisplay" style="margin-left:auto;font-size:11px;color:var(--text-dimmer)"></span>
      </div>
      <div class="progress-bar" id="algorithmProgressBar" style="display:none">
        <div class="progress-fill" id="algorithmProgressFill" style="width:0%"></div>
      </div>
      <div id="algorithmResultsContent">
        <div class="empty-state" id="algorithmEmpty">
          <div class="icon">
            <svg width="60" height="60" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" stroke-linecap="round" stroke-linejoin="round" style="opacity:0.3">
              <path d="M12 20V10"/><path d="M18 20V4"/><path d="M6 20v-4"/>
            </svg>
          </div>
          <p>> upload a video to analyze for social media algorithms _</p>
        </div>
      </div>
    </div>
  </div><!-- /panelAlgorithm -->

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
window._themeMatrixFill = '#00ff41';
window._themeMatrixBg = 'rgba(0, 10, 0, 0.05)';
function drawMatrix() {
  ctx.fillStyle = window._themeMatrixBg;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = window._themeMatrixFill;
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
  const elId = panel === 'text' ? 'textTimerDisplay' : panel === 'transcribe' ? 'transcribeTimerDisplay' : panel === 'algorithm' ? 'algorithmTimerDisplay' : 'audioTimerDisplay';
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

// ==================== THEME SWITCHER ====================
const THEME_COLORS = {
  matrix: { stroke: '#00ff41', fill: '#00ff41', glint: '#aaffaa',
    cursor: "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='28' height='18' viewBox='0 0 28 18'%3E%3Cpath d='M1 9 Q14 0 27 9 Q14 18 1 9 Z' fill='none' stroke='%2300ff41' stroke-width='1.5'/%3E%3Ccircle cx='14' cy='9' r='4' fill='none' stroke='%2300ff41' stroke-width='1.2'/%3E%3Ccircle cx='14' cy='9' r='1.5' fill='%2300ff41'/%3E%3C/svg%3E\") 14 9, auto",
    matrixFill: '#00ff41', matrixBg: 'rgba(0, 10, 0, 0.05)' },
  glass: { stroke: '#2a6cb6', fill: '#2a6cb6', glint: '#6eaadf',
    cursor: "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='28' height='18' viewBox='0 0 28 18'%3E%3Cpath d='M1 9 Q14 0 27 9 Q14 18 1 9 Z' fill='none' stroke='%232a6cb6' stroke-width='1.5'/%3E%3Ccircle cx='14' cy='9' r='4' fill='none' stroke='%232a6cb6' stroke-width='1.2'/%3E%3Ccircle cx='14' cy='9' r='1.5' fill='%232a6cb6'/%3E%3C/svg%3E\") 14 9, auto",
    matrixFill: '#2a6cb6', matrixBg: 'rgba(232, 236, 240, 0.05)' },
};

function applyTheme(theme) {
  const t = THEME_COLORS[theme] || THEME_COLORS.matrix;
  document.body.classList.toggle('glass', theme === 'glass');
  // Update canvas opacity
  const c = document.getElementById('matrixCanvas');
  if (c) c.style.opacity = theme === 'glass' ? '0' : '0.12';
  // Update cursor globally
  document.querySelectorAll('*').forEach(el => { el.style.cursor = t.cursor; });
  // Inject a style for cursor inheritance
  let cursorStyle = document.getElementById('cursorOverride');
  if (!cursorStyle) {
    cursorStyle = document.createElement('style');
    cursorStyle.id = 'cursorOverride';
    document.head.appendChild(cursorStyle);
  }
  cursorStyle.textContent = '* { cursor: ' + t.cursor + ' !important; }';
  // Update SVG logo colors
  document.querySelectorAll('.logo-stroke').forEach(el => el.setAttribute('stroke', t.stroke));
  document.querySelectorAll('.logo-fill').forEach(el => el.setAttribute('fill', t.fill));
  document.querySelectorAll('.logo-glint').forEach(el => el.setAttribute('fill', t.glint));
  // Update label
  const label = document.getElementById('themeLabel');
  if (label) label.textContent = theme.toUpperCase();
  // Update matrix rain colors for glass
  window._themeMatrixFill = t.matrixFill;
  window._themeMatrixBg = t.matrixBg;
  // Update menu active states
  document.querySelectorAll('.theme-option').forEach(opt => {
    opt.classList.toggle('active', opt.dataset.theme === theme);
  });
  localStorage.setItem('ss-theme', theme);
}
function toggleThemeMenu(e) {
  e.stopPropagation();
  const sel = document.getElementById('themeSelector');
  if (sel) sel.classList.toggle('open');
}
function selectTheme(theme) {
  applyTheme(theme);
  const sel = document.getElementById('themeSelector');
  if (sel) sel.classList.remove('open');
}
// Close menu on outside click
document.addEventListener('click', function(e) {
  const sel = document.getElementById('themeSelector');
  if (sel && !sel.contains(e.target)) sel.classList.remove('open');
});
// ==================== ALGORITHM TAB ====================
let algorithmUploadedFile = null;
const VIDEO_EXTS = ['.mp4','.mov','.avi','.mkv','.webm','.m4v','.flv','.wmv'];

(function setupAlgorithmDrop() {
  const zone = document.getElementById('algorithmDropZone');
  const input = document.getElementById('algorithmFileInput');
  if (!zone || !input) return;
  ['dragenter','dragover'].forEach(e => zone.addEventListener(e, ev => {
    ev.preventDefault(); zone.classList.add('dragover');
  }));
  ['dragleave','drop'].forEach(e => zone.addEventListener(e, ev => {
    ev.preventDefault(); zone.classList.remove('dragover');
  }));
  zone.addEventListener('drop', ev => { if (ev.dataTransfer.files[0]) handleAlgorithmFile(ev.dataTransfer.files[0]); });
  input.addEventListener('change', () => { if (input.files[0]) handleAlgorithmFile(input.files[0]); });
})();

function handleAlgorithmFile(file) {
  const ext = '.' + file.name.split('.').pop().toLowerCase();
  if (!VIDEO_EXTS.includes(ext)) {
    alert('Please upload a video file (' + VIDEO_EXTS.join(', ') + ')');
    return;
  }
  if (file.size > 500 * 1024 * 1024) {
    alert('File too large. Maximum 500MB.');
    return;
  }
  algorithmUploadedFile = file;
  const sizeMB = (file.size / (1024*1024)).toFixed(1);
  document.getElementById('algorithmUploadText').innerHTML = '> ' + file.name + ' <span style="color:var(--text-dimmer)">(' + sizeMB + ' MB)</span>';
  document.getElementById('algorithmUploadHint').textContent = 'Click to change file';
  document.getElementById('algorithmDropZone').onclick = function() { document.getElementById('algorithmFileInput').click(); };
  document.getElementById('algorithmScanBtn').disabled = false;
  // Reset results
  document.getElementById('algorithmResultsContent').innerHTML = '<div class="empty-state" id="algorithmEmpty"><div class="icon"><svg width="60" height="60" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1" stroke-linecap="round" stroke-linejoin="round" style="opacity:0.3"><path d="M12 20V10"/><path d="M18 20V4"/><path d="M6 20v-4"/></svg></div><p>> ready to analyze _</p></div>';
}

async function startAlgorithm() {
  if (!algorithmUploadedFile) return;
  const btn = document.getElementById('algorithmScanBtn');
  const status = document.getElementById('algorithmStatus');
  const statusText = document.getElementById('algorithmStatusText');
  const progressBar = document.getElementById('algorithmProgressBar');
  const progressFill = document.getElementById('algorithmProgressFill');
  const resultsDiv = document.getElementById('algorithmResultsContent');

  btn.disabled = true;
  status.style.display = 'flex';
  progressBar.style.display = 'block';
  progressFill.style.width = '0%';
  statusText.textContent = 'Uploading video...';
  resultsDiv.innerHTML = '';
  startAudioTimer('algorithm');

  const fd = new FormData();
  fd.append('video', algorithmUploadedFile);

  try {
    const resp = await fetch('/algorithm/scan', { method: 'POST', body: fd });
    const data = await resp.json();
    if (data.error) { throw new Error(data.error); }
    const scanId = data.scan_id;

    function poll() {
      fetch('/algorithm/scan/progress?id=' + scanId).then(r => r.json()).then(prog => {
        if (prog.error) {
          statusText.textContent = 'Error: ' + prog.error;
          document.getElementById('algorithmSpinner').style.display = 'none';
          cancelAnimationFrame(audioTimerRAF);
          btn.disabled = false;
          return;
        }
        if (prog.done) {
          cancelAnimationFrame(audioTimerRAF);
          status.style.display = 'none';
          progressBar.style.display = 'none';
          btn.disabled = false;
          renderAlgorithmResults(prog);
          return;
        }
        progressFill.style.width = (prog.percent || 0) + '%';
        statusText.textContent = prog.phase_detail || 'Analyzing...';
        setTimeout(poll, 500);
      }).catch(err => {
        statusText.textContent = 'Error: ' + err.message;
        cancelAnimationFrame(audioTimerRAF);
        btn.disabled = false;
      });
    }
    poll();
  } catch (err) {
    statusText.textContent = 'Error: ' + err.message;
    document.getElementById('algorithmSpinner').style.display = 'none';
    cancelAnimationFrame(audioTimerRAF);
    btn.disabled = false;
  }
}

function algoScoreColor(score) {
  if (score >= 85) return '#00ff41';
  if (score >= 70) return '#7fff00';
  if (score >= 50) return '#ffaa00';
  if (score >= 30) return '#ff6e41';
  return '#ff4141';
}

function algoScoreColorGlass(score) {
  if (score >= 85) return '#1a8a3e';
  if (score >= 70) return '#5a9a10';
  if (score >= 50) return '#b87a00';
  if (score >= 30) return '#c44';
  return '#a00';
}

function algoBarColor(score) {
  const isGlass = document.body.classList.contains('glass');
  return isGlass ? algoScoreColorGlass(score) : algoScoreColor(score);
}

function renderScoreRing(score, size) {
  size = size || 80;
  const r = (size - 8) / 2;
  const circ = 2 * Math.PI * r;
  const offset = circ * (1 - score / 100);
  const isGlass = document.body.classList.contains('glass');
  const color = isGlass ? algoScoreColorGlass(score) : algoScoreColor(score);
  const trackColor = isGlass ? 'rgba(0,0,0,0.08)' : 'rgba(255,255,255,0.06)';
  return '<svg width="' + size + '" height="' + size + '" viewBox="0 0 ' + size + ' ' + size + '">' +
    '<circle cx="' + (size/2) + '" cy="' + (size/2) + '" r="' + r + '" fill="none" stroke="' + trackColor + '" stroke-width="6"/>' +
    '<circle cx="' + (size/2) + '" cy="' + (size/2) + '" r="' + r + '" fill="none" stroke="' + color + '" stroke-width="6" ' +
    'stroke-dasharray="' + circ + '" stroke-dashoffset="' + offset + '" stroke-linecap="round" ' +
    'transform="rotate(-90 ' + (size/2) + ' ' + (size/2) + ')" style="transition:stroke-dashoffset 1s ease;filter:drop-shadow(0 0 4px ' + color + ')"/>' +
    '</svg>';
}

function fmtDuration(s) {
  if (s < 60) return s.toFixed(1) + 's';
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  if (m < 60) return m + ':' + (sec < 10 ? '0' : '') + sec;
  const h = Math.floor(m / 60);
  return h + ':' + ((m%60) < 10 ? '0' : '') + (m%60) + ':' + (sec < 10 ? '0' : '') + sec;
}

function renderAlgorithmResults(data) {
  const p = data.properties || {};
  const scores = data.scores || {};
  let html = '';

  // Header
  html += '<div class="results-header" style="margin-bottom:16px;padding:12px 16px;background:var(--bg-panel);border:1px solid var(--border);border-radius:8px">';
  html += '<div style="font-size:14px;font-weight:600;color:var(--accent);margin-bottom:4px">' + (data.filename || 'Video') + '</div>';
  html += '<div style="font-size:11px;color:var(--text-dimmer)">Analyzed in ' + (data.elapsed || 0) + 's</div>';
  html += '</div>';

  // Video Properties
  html += '<div style="font-size:11px;color:var(--text-dimmer);text-transform:uppercase;letter-spacing:2px;margin-bottom:8px">> Video Properties</div>';
  html += '<div class="algo-props">';
  const props = [
    ['Duration', fmtDuration(p.duration || 0)],
    ['Resolution', (p.width||0) + ' x ' + (p.height||0)],
    ['Aspect Ratio', (p.aspect_ratio||0) + ' (' + (p.orientation||'?') + ')'],
    ['Frame Rate', (p.fps||0) + ' fps'],
    ['Bitrate', (p.bitrate_kbps||0) + ' kbps'],
    ['File Size', (p.file_size_mb||0) + ' MB'],
    ['Audio', p.has_audio ? 'Yes (loudness: ' + (p.audio_loudness||0) + ')' : 'No audio'],
    ['Scene Changes', (p.scene_changes||0) + ' (' + (p.scene_change_rate||0) + '/min)'],
    ['Faces Detected', p.face_detected ? 'Yes (' + Math.round((p.face_frequency||0)*100) + '% of frames)' : 'No'],
    ['Text Overlays', p.has_text_overlay ? 'Yes' : 'No'],
    ['Hook Strength', (p.hook_strength||0) + '/100'],
    ['Brightness', Math.round(p.avg_brightness||0) + '/255'],
  ];
  props.forEach(function(pr) {
    html += '<div class="algo-prop"><span class="prop-label">' + pr[0] + '</span><span class="prop-value">' + pr[1] + '</span></div>';
  });
  html += '</div>';

  // Score Cards Overview
  html += '<div style="font-size:11px;color:var(--text-dimmer);text-transform:uppercase;letter-spacing:2px;margin-bottom:8px">> Platform Scores</div>';
  // Sort by score descending
  const sorted = Object.keys(scores).sort(function(a,b) { return scores[b].score - scores[a].score; });
  html += '<div class="algo-summary">';
  sorted.forEach(function(key) {
    const s = scores[key];
    html += '<div class="algo-card" onclick="document.getElementById(\'detail-' + key + '\').scrollIntoView({behavior:\'smooth\',block:\'center\'})">';
    html += '<div class="card-icon">' + s.icon + '</div>';
    html += '<div class="card-name">' + s.name + '</div>';
    html += '<div class="card-ring">' + renderScoreRing(s.score, 80) + '</div>';
    html += '<div class="card-score grade-' + s.grade.toLowerCase() + '">' + s.score + '</div>';
    html += '<div class="card-grade grade-' + s.grade.toLowerCase() + '">' + s.grade + '</div>';
    html += '</div>';
  });
  html += '</div>';

  // Detailed Breakdowns
  html += '<div style="font-size:11px;color:var(--text-dimmer);text-transform:uppercase;letter-spacing:2px;margin-bottom:8px">> Detailed Breakdown</div>';
  html += '<div class="algo-details">';
  const factorLabels = { duration:'Duration', aspect_ratio:'Aspect Ratio', resolution:'Resolution', audio:'Audio', engagement:'Engagement', platform_bonus:'Platform Fit' };
  sorted.forEach(function(key) {
    const s = scores[key];
    html += '<div class="algo-detail" id="detail-' + key + '">';
    html += '<div class="algo-detail-header">';
    html += '<span class="detail-icon">' + s.icon + '</span>';
    html += '<span class="detail-name">' + s.name + '</span>';
    html += '<span class="detail-badge grade-' + s.grade.toLowerCase() + '">' + s.score + ' ' + s.grade + '</span>';
    html += '</div>';
    html += '<div class="algo-breakdown">';
    Object.keys(factorLabels).forEach(function(f) {
      const val = (s.breakdown && s.breakdown[f]) || 0;
      const color = algoBarColor(val);
      html += '<div class="algo-bar-row">';
      html += '<span class="algo-bar-label">' + factorLabels[f] + '</span>';
      html += '<div class="algo-bar-track"><div class="algo-bar-fill" style="width:' + val + '%;background:' + color + '"></div></div>';
      html += '<span class="algo-bar-val">' + val + '</span>';
      html += '</div>';
    });
    html += '</div>';
    if (s.tips && s.tips.length > 0) {
      html += '<ul class="algo-tips">';
      s.tips.forEach(function(tip) { html += '<li>' + tip + '</li>'; });
      html += '</ul>';
    } else {
      html += '<div class="algo-no-tips">> Great match for ' + s.name + '!</div>';
    }
    html += '</div>';
  });
  html += '</div>';

  document.getElementById('algorithmResultsContent').innerHTML = html;
}

// Check if OpenCV is available for Algorithm tab
(async function checkCv2Status() {
  try {
    const resp = await fetch('/algorithm/status');
    const data = await resp.json();
    if (!data.available) {
      const notice = document.getElementById('algorithmDisabledNotice');
      if (notice) notice.hidden = false;
      const btn = document.getElementById('algorithmScanBtn');
      if (btn) btn.style.display = 'none';
    }
  } catch(e) {}
})();

// Apply saved theme on load
applyTheme(localStorage.getItem('ss-theme') || 'matrix');

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
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Inter:wght@300;400;500&display=swap');

  :root {
    --bg: #000a00; --bg-panel: rgba(0, 10, 0, 0.9); --accent: #00ff41;
    --accent-dim: #0a6e2a; --accent-dimmer: #0a3e1a; --accent-faint: #0a4e1a;
    --text: #00ff41; --text-dim: #0a8e3a; --text-dimmer: #0a5e2a;
    --border: #0a6e2a; --error: #ff6e41;
    --glow-shadow: rgba(0,255,65,0.15); --glow-strong: rgba(0,255,65,0.3);
    --glow-text: 0 0 10px #00ff41, 0 0 30px #008f11;
    --scanline-bg: repeating-linear-gradient(0deg, rgba(0,0,0,0.15) 0px, rgba(0,0,0,0.15) 1px, transparent 1px, transparent 3px);
    --font: 'Share Tech Mono', 'Courier New', monospace;
    --login-bg: rgba(0, 10, 0, 0.9); --login-shadow: 0 0 40px rgba(0,255,65,0.15), 0 0 80px rgba(0,255,65,0.05);
  }
  body.glass {
    --bg: #e8ecf0; --bg-panel: rgba(255,255,255,0.45); --accent: #2a6cb6;
    --accent-dim: rgba(42,108,182,0.55); --accent-dimmer: rgba(42,108,182,0.2);
    --accent-faint: rgba(42,108,182,0.08);
    --text: #1a2a3a; --text-dim: rgba(30,50,70,0.7); --text-dimmer: rgba(60,80,110,0.5);
    --border: rgba(0,0,0,0.1); --error: #d44;
    --glow-shadow: rgba(42,108,182,0.08); --glow-strong: rgba(42,108,182,0.15);
    --glow-text: 0 0 8px rgba(42,108,182,0.25);
    --scanline-bg: none;
    --font: 'Inter', 'Share Tech Mono', sans-serif;
    --login-bg: rgba(255,255,255,0.45); --login-shadow: 0 2px 30px rgba(0,0,0,0.08), 0 1px 3px rgba(0,0,0,0.06);
  }
  body.glass .login-wrap::before { display: none; }
  body.glass .login-box { backdrop-filter: blur(30px); -webkit-backdrop-filter: blur(30px);
    background: rgba(255,255,255,0.45);
    border-top: 1px solid rgba(255,255,255,0.7);
    border: 1px solid rgba(255,255,255,0.6);
    box-shadow: var(--login-shadow); }
  body.glass .login-box h1 { animation: none; font-family: 'Inter', sans-serif; font-weight: 300; letter-spacing: 6px; color: #1a2a3a; text-shadow: none; }
  body.glass .login-box .subtitle { font-family: 'Inter', sans-serif; color: rgba(30,50,70,0.6); }
  body.glass .login-btn { color: #2a6cb6; border-color: rgba(42,108,182,0.4); }
  body.glass .login-btn:hover { background: #2a6cb6; color: #fff; box-shadow: 0 0 15px rgba(42,108,182,0.3); }
  body.glass .lock-icon { color: rgba(60,80,110,0.5); }
  body.glass::after { content:''; position:fixed; top:0; left:0; width:100%; height:100%;
    background: radial-gradient(ellipse at 30% 30%, rgba(42,108,182,0.06) 0%, transparent 50%),
                radial-gradient(ellipse at 70% 70%, rgba(100,60,200,0.03) 0%, transparent 50%),
                radial-gradient(ellipse at 50% 50%, rgba(255,255,255,0.4) 0%, transparent 70%);
    pointer-events:none; z-index:0; }

  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: var(--font); background: var(--bg); color: var(--text);
         min-height: 100vh; display: flex; align-items: center; justify-content: center;
         position: relative; overflow: hidden; }
  #matrixCanvas { position: fixed; top: 0; left: 0; width: 100%; height: 100%;
                  z-index: 0; pointer-events: none; opacity: 0.15; transition: opacity 0.5s; }
  .login-wrap { position: relative; z-index: 1; }
  .login-wrap::before { content: ''; position: fixed; top: 0; left: 0; width: 100%; height: 100%;
    background: var(--scanline-bg); pointer-events: none; z-index: 9999; }
  .login-box { background: var(--login-bg); border: 2px solid var(--accent); border-radius: 8px;
    padding: 40px 35px; width: 380px; max-width: 90vw; text-align: center;
    box-shadow: var(--login-shadow); backdrop-filter: blur(8px); }
  .login-box .logo-svg { width: 64px; height: 64px; margin-bottom: 15px;
    filter: drop-shadow(0 0 10px var(--accent)); }
  .login-box h1 { font-size: 20px; letter-spacing: 3px; text-transform: uppercase;
    text-shadow: var(--glow-text); margin-bottom: 6px; }
  .login-box .subtitle { color: var(--accent-dim); font-size: 12px; margin-bottom: 28px; letter-spacing: 1px; }
  .login-box label { display: block; text-align: left; color: var(--text-dim); font-size: 11px;
    text-transform: uppercase; letter-spacing: 1px; margin-bottom: 6px; }
  .login-box input[type="password"] { width: 100%; padding: 12px 14px; border-radius: 4px;
    border: 1px solid var(--border); background: rgba(0, 5, 0, 0.9); color: var(--text);
    font-size: 18px; font-family: var(--font); text-align: center;
    letter-spacing: 6px; margin-bottom: 20px; }
  body.glass .login-box input[type="password"] { background: rgba(255,255,255,0.5); color: #1a2a3a; border-color: rgba(0,0,0,0.12); }
  body.glass .login-box input[type="password"]:focus { border-color: #2a6cb6; box-shadow: 0 0 10px rgba(42,108,182,0.2); }
  body.glass .login-box input[type="password"]::placeholder { color: rgba(60,80,110,0.4); }
  body.glass .login-box label { color: rgba(30,50,70,0.6); }
  .login-box input[type="password"]:focus { outline: none; border-color: var(--accent);
    box-shadow: 0 0 15px var(--glow-strong); }
  .login-box input[type="password"]::placeholder { color: var(--accent-dimmer); letter-spacing: 2px; font-size: 13px; }
  .login-btn { width: 100%; padding: 14px; background: transparent; color: var(--accent);
    border: 2px solid var(--accent); border-radius: 4px; font-size: 14px; font-weight: 600;
    cursor: pointer; text-transform: uppercase; letter-spacing: 3px;
    font-family: var(--font); transition: all 0.3s; }
  .login-btn:hover { background: var(--accent); color: var(--bg);
    box-shadow: 0 0 20px var(--accent), 0 0 40px var(--glow-strong); }
  .error-msg { color: var(--error); font-size: 12px; margin-top: 14px; min-height: 16px;
    text-shadow: 0 0 6px rgba(255,110,65,0.5); }
  body.glass .error-msg { color: #d44; text-shadow: none; }
  @keyframes flicker { 0%, 95% { opacity: 1; } 96% { opacity: 0.8; } 97% { opacity: 1; }
    98% { opacity: 0.6; } 100% { opacity: 1; } }
  .login-box h1 { animation: flicker 4s infinite; }
  .lock-icon { font-size: 10px; color: var(--text-dimmer); margin-top: 18px; }

  .theme-selector-login { position: fixed; top: 20px; right: 20px; z-index: 10001; }
  .theme-selector-login .theme-btn {
    background: rgba(0,10,0,0.6); border: 1px solid var(--accent-dimmer); color: var(--accent-dim);
    padding: 6px 14px; border-radius: 20px; cursor: pointer; font-family: var(--font);
    font-size: 11px; letter-spacing: 1px; transition: all 0.3s; display: flex; align-items: center; gap: 6px;
    backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px); }
  body.glass .theme-selector-login .theme-btn { background: rgba(255,255,255,0.5); border-color: rgba(0,0,0,0.1); color: #2a6cb6; }
  body.glass .theme-selector-login .theme-btn:hover { border-color: #2a6cb6; box-shadow: 0 0 8px rgba(42,108,182,0.15); }
  .theme-selector-login .theme-btn:hover { border-color: var(--accent); color: var(--accent);
    box-shadow: 0 0 12px var(--glow-shadow); }
  .theme-selector-login .theme-dot { width: 8px; height: 8px; border-radius: 50%;
    background: var(--accent); box-shadow: 0 0 6px var(--accent); transition: all 0.3s; }
  .theme-selector-login .arrow { font-size: 8px; margin-left: 2px; transition: transform 0.2s; }
  .theme-selector-login.open .arrow { transform: rotate(180deg); }
  .theme-menu { position: absolute; top: calc(100% + 8px); right: 0; min-width: 160px;
    background: var(--bg-panel); border: 1px solid var(--border); border-radius: 8px;
    backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4); padding: 6px 0;
    opacity: 0; visibility: hidden; transform: translateY(-6px);
    transition: all 0.2s ease; z-index: 100; }
  body.glass .theme-menu { background: rgba(255,255,255,0.75); box-shadow: 0 8px 32px rgba(0,0,0,0.1); border-color: rgba(0,0,0,0.08); }
  body.glass .theme-option { color: rgba(30,50,70,0.6); }
  body.glass .theme-option:hover { background: rgba(42,108,182,0.06); color: #2a6cb6; }
  body.glass .theme-option.active { color: #2a6cb6; }
  body.glass .theme-option.active .opt-dot { box-shadow: 0 0 6px rgba(42,108,182,0.4); }
  .theme-selector-login.open .theme-menu { opacity: 1; visibility: visible; transform: translateY(0); }
  .theme-option { display: flex; align-items: center; gap: 10px; padding: 10px 16px;
    cursor: pointer; transition: background 0.15s; font-family: var(--font);
    font-size: 12px; color: var(--accent-dim); letter-spacing: 1px; }
  .theme-option:hover { background: var(--hover-bg); color: var(--accent); }
  .theme-option.active { color: var(--accent); }
  .theme-option .opt-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0;
    border: 2px solid currentColor; transition: all 0.2s; }
  .theme-option.active .opt-dot { background: var(--accent); box-shadow: 0 0 8px var(--accent); }
  .theme-option .opt-label { flex: 1; }
  .theme-option .opt-check { font-size: 14px; opacity: 0; transition: opacity 0.2s; }
  .theme-option.active .opt-check { opacity: 1; }
</style>
</head>
<body>
<canvas id="matrixCanvas"></canvas>
<div class="theme-selector-login" id="themeSelector">
  <button class="theme-btn" onclick="toggleThemeMenu(event)" title="Switch Theme">
    <span class="theme-dot"></span> <span id="themeLabel">MATRIX</span> <span class="arrow">&#9660;</span>
  </button>
  <div class="theme-menu" id="themeMenu">
    <div class="theme-option active" data-theme="matrix" onclick="selectTheme('matrix')">
      <span class="opt-dot" style="color:#00ff41;"></span>
      <span class="opt-label">MATRIX</span>
      <span class="opt-check">&#10003;</span>
    </div>
    <div class="theme-option" data-theme="glass" onclick="selectTheme('glass')">
      <span class="opt-dot" style="color:#2a6cb6;"></span>
      <span class="opt-label">GLASS</span>
      <span class="opt-check">&#10003;</span>
    </div>
  </div>
</div>
<div class="login-wrap">
<div class="login-box">
  <svg class="logo-svg" id="loginLogoSvg" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
    <circle cx="40" cy="40" r="28" fill="none" stroke="#00ff41" stroke-width="4" class="logo-stroke"/>
    <line x1="60" y1="60" x2="88" y2="88" stroke="#00ff41" stroke-width="6" stroke-linecap="round" class="logo-stroke"/>
    <path d="M22 40 Q40 25 58 40 Q40 55 22 40 Z" fill="none" stroke="#00ff41" stroke-width="2.5" class="logo-stroke"/>
    <circle id="loginIris" cx="40" cy="40" r="7" fill="none" stroke="#00ff41" stroke-width="2" class="logo-stroke"/>
    <circle id="loginPupil" cx="40" cy="40" r="3" fill="#00ff41" class="logo-fill"/>
    <circle id="loginGlint" cx="37" cy="37" r="1.5" fill="#aaffaa" class="logo-glint"/>
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
window._themeMatrixFill = '#00ff41';
window._themeMatrixBg = 'rgba(0, 10, 0, 0.05)';
function drawMatrix() {
  ctx.fillStyle = window._themeMatrixBg;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = window._themeMatrixFill;
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

// Theme switcher for login page
const THEME_COLORS = {
  matrix: { stroke: '#00ff41', fill: '#00ff41', glint: '#aaffaa',
    cursor: "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='28' height='18' viewBox='0 0 28 18'%3E%3Cpath d='M1 9 Q14 0 27 9 Q14 18 1 9 Z' fill='none' stroke='%2300ff41' stroke-width='1.5'/%3E%3Ccircle cx='14' cy='9' r='4' fill='none' stroke='%2300ff41' stroke-width='1.2'/%3E%3Ccircle cx='14' cy='9' r='1.5' fill='%2300ff41'/%3E%3C/svg%3E\") 14 9, auto",
    matrixFill: '#00ff41', matrixBg: 'rgba(0, 10, 0, 0.05)' },
  glass: { stroke: '#2a6cb6', fill: '#2a6cb6', glint: '#6eaadf',
    cursor: "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='28' height='18' viewBox='0 0 28 18'%3E%3Cpath d='M1 9 Q14 0 27 9 Q14 18 1 9 Z' fill='none' stroke='%232a6cb6' stroke-width='1.5'/%3E%3Ccircle cx='14' cy='9' r='4' fill='none' stroke='%232a6cb6' stroke-width='1.2'/%3E%3Ccircle cx='14' cy='9' r='1.5' fill='%232a6cb6'/%3E%3C/svg%3E\") 14 9, auto",
    matrixFill: '#2a6cb6', matrixBg: 'rgba(232, 236, 240, 0.05)' },
};
function applyTheme(theme) {
  const t = THEME_COLORS[theme] || THEME_COLORS.matrix;
  document.body.classList.toggle('glass', theme === 'glass');
  const c = document.getElementById('matrixCanvas');
  if (c) c.style.opacity = theme === 'glass' ? '0' : '0.15';
  let cursorStyle = document.getElementById('cursorOverride');
  if (!cursorStyle) { cursorStyle = document.createElement('style'); cursorStyle.id = 'cursorOverride'; document.head.appendChild(cursorStyle); }
  cursorStyle.textContent = '* { cursor: ' + t.cursor + ' !important; }';
  document.querySelectorAll('.logo-stroke').forEach(el => el.setAttribute('stroke', t.stroke));
  document.querySelectorAll('.logo-fill').forEach(el => el.setAttribute('fill', t.fill));
  document.querySelectorAll('.logo-glint').forEach(el => el.setAttribute('fill', t.glint));
  const label = document.getElementById('themeLabel');
  if (label) label.textContent = theme === 'glass' ? 'GLASS' : 'MATRIX';
  // Update dropdown active states
  document.querySelectorAll('.theme-option').forEach(opt => {
    opt.classList.toggle('active', opt.getAttribute('data-theme') === theme);
  });
  window._themeMatrixFill = t.matrixFill;
  window._themeMatrixBg = t.matrixBg;
  localStorage.setItem('ss-theme', theme);
}
function toggleThemeMenu(e) {
  e.stopPropagation();
  document.getElementById('themeSelector').classList.toggle('open');
}
function selectTheme(theme) {
  applyTheme(theme);
  document.getElementById('themeSelector').classList.remove('open');
}
document.addEventListener('click', function(e) {
  const sel = document.getElementById('themeSelector');
  if (sel && !sel.contains(e.target)) sel.classList.remove('open');
});
applyTheme(localStorage.getItem('ss-theme') || 'matrix');
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


# ==================== ALGORITHM ROUTES ====================
@app.route('/algorithm/status')
def algorithm_status():
    """Check if video analysis (OpenCV) is available."""
    return jsonify(available=HAS_CV2)


@app.route('/algorithm/scan', methods=['POST'])
def algorithm_scan():
    if not HAS_CV2:
        return jsonify(error="Video analysis not available. Install opencv-python."), 501
    if 'video' not in request.files:
        return jsonify(error="No video file uploaded"), 400
    file = request.files['video']
    if not file.filename:
        return jsonify(error="No file selected"), 400
    ext = Path(file.filename).suffix.lower()
    if ext not in VIDEO_EXTENSIONS:
        return jsonify(error="Please upload a video file"), 400
    temp_path = os.path.join(ALGORITHM_UPLOAD_DIR, f"{uuid.uuid4().hex}{ext}")
    file.save(temp_path)
    scan_id = uuid.uuid4().hex[:12]
    original_filename = file.filename or 'unknown'
    with _algorithm_lock:
        _algorithm_progress[scan_id] = {
            'phase': 'starting', 'phase_detail': 'Initializing...',
            'percent': 0, 'done': False, 'error': None, 'result': None,
            'start_time': time.time(), 'filename': original_filename,
        }

    def run_analysis():
        try:
            analyzer = VideoAnalyzer(temp_path)
            result = analyzer.analyze(scan_id=scan_id)
            elapsed = time.time() - _algorithm_progress[scan_id]['start_time']
            result['elapsed'] = round(elapsed, 1)
            result['filename'] = original_filename
            with _algorithm_lock:
                _algorithm_progress[scan_id]['done'] = True
                _algorithm_progress[scan_id]['result'] = result
        except Exception as e:
            _sec_log.error(f"Algorithm scan error: {e}")
            with _algorithm_lock:
                _algorithm_progress[scan_id]['done'] = True
                _algorithm_progress[scan_id]['error'] = str(e)
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    threading.Thread(target=run_analysis, daemon=True).start()
    return jsonify(scan_id=scan_id)


@app.route('/algorithm/scan/progress')
def algorithm_scan_progress():
    scan_id = request.args.get('id', '')
    with _algorithm_lock:
        prog = _algorithm_progress.get(scan_id)
    if not prog:
        return jsonify(error="Unknown scan"), 404
    if prog['done']:
        with _algorithm_lock:
            result = _algorithm_progress.pop(scan_id, {})
        if result.get('error'):
            return jsonify(error=result['error'], done=True), 500
        return jsonify(done=True, **(result.get('result') or {}))
    return jsonify(
        done=False, phase=prog['phase'],
        phase_detail=prog['phase_detail'], percent=prog['percent'],
    )


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
