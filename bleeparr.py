#!/usr/bin/env python3

# -----------------------------------------
# Imports
# -----------------------------------------

import argparse
import os
import subprocess
import srt
import json
import re
import shutil
import time
from faster_whisper import WhisperModel
from datetime import timedelta
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
import tempfile
import errno

# -----------------------------------------
# CLI Parsing
# -----------------------------------------

parser = argparse.ArgumentParser(description="Bleeparr: Automated Profanity Censorship Tool")

parser.add_argument(
    '--beep', 
    action='store_true', 
    help='Insert a tone/beep instead of mute'
)

parser.add_argument(
    '--beep-mode', 
    choices=['words', 'segments', 'both'], 
    help='Choose beep mode: only words found by Whisper, full subtitle segments, or both (default: words)'
)

parser.add_argument(
    '--temp-dir', 
    type=str, 
    help='Custom temporary directory for storing clips'
)

parser.add_argument(
    '--retain-clips', 
    action='store_true', 
    help='Retain clips folder and files after processing (default: delete)'
)

parser.add_argument("--input", required=True, help="Input video file path")
parser.add_argument("--subtitle", help="Optional subtitle file path (.srt)")
parser.add_argument("--boost-db", type=int, default=6, help="Audio boost in dB when extracting clips")
# removed for S,M,FSM switches parser.add_argument("--fallback-subtitle-mute", action="store_true", help="Mute full subtitle section if Whisper fails")
# removed for S,M,FSM switches parser.add_argument("--whisper-tiered", action="store_true", help="First try Whisper small, fallback to medium")
# new S,M,FSM bleeptool
parser.add_argument(
    "--bleeptool",
    default="S-M-FSM",
    help=(
        "What passes to run: S (Whisper small), "
        "M (Whisper medium fallback), FSM (full-subtitle mute). "
        "Combine with dashes: e.g. 'S', 'S-M', 'S-M-FSM', 'FSM'."
    )
)
parser.add_argument("--pre-buffer", type=int, default=100, help="Pre-buffer mute milliseconds (default 100)")
parser.add_argument("--post-buffer", type=int, default=100, help="Post-buffer mute milliseconds (default 100)")
parser.add_argument("--output-suffix", default="(edited by Bleeparr)", help="Suffix to append to output filename")
parser.add_argument("--delete-original", action="store_true", help="Delete original video after successful censoring")
parser.add_argument("--no-keep-subs", action="store_true", help="Delete subtitle file after processing")
parser.add_argument("--swears", default="swears.txt", help="File containing swear words to censor")
parser.add_argument("--alert-censoring-off", action="store_true", help="Disable replacing bad‐words with asterisks in all alerts (prints the raw word)")
parser.add_argument(
    "--subtitle-lang",
    default="eng",
    help="Preferred subtitle language (e.g., eng, spa, fra). Default: eng"
)
parser.add_argument(
    "--no-embedded-subs",
    action="store_true",
    help="Skip checking/extracting embedded subtitles"
)

parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Analyze and plan without writing an edited file or deleting anything"
)

args = parser.parse_args()

DRY_RUN = args.dry_run
use_beep = args.beep
beep_mode = args.beep_mode if args.beep_mode else 'words' if use_beep else None
if not use_beep and args.beep_mode:
    print("⚠️ Warning: --beep-mode has no effect unless --beep is also set.")
custom_temp_dir = args.temp_dir
retain_clips = args.retain_clips

BLEEPTOOL = [p.strip().upper() for p in args.bleeptool.split("-")]
DO_S   = "S"   in BLEEPTOOL
DO_M   = "M"   in BLEEPTOOL
DO_FSM = "FSM" in BLEEPTOOL


# -----------------------------------------
# Settings (from CLI)
# -----------------------------------------

INPUT_VIDEO = args.input
INPUT_SRT = args.subtitle
BOOST_DB = args.boost_db
# FALLBACK_SUBTITLE_MUTE = args.fallback_subtitle_mute
# TIERED_WHISPER = args.whisper_tiered
DO_SMALL_MODEL  = DO_S
DO_MEDIUM_MODEL = DO_M
DO_FULL_SUB_MUTE= DO_FSM
PRE_BUFFER_MS = args.pre_buffer
POST_BUFFER_MS = args.post_buffer
OUTPUT_SUFFIX = args.output_suffix
DELETE_ORIGINAL = args.delete_original
KEEP_SUBS = not args.no_keep_subs
SWEARS_FILE = args.swears
ALERT_CENSORING_OFF = args.alert_censoring_off

if custom_temp_dir:
    base_clips_path = os.path.abspath(custom_temp_dir)
    try:
        os.makedirs(base_clips_path, exist_ok=True)
    except Exception as e:
        print(f"❌ Failed to create or access --temp-dir: {base_clips_path}")
        print(f"Error: {e}")
        exit(1)
    used_custom_temp_dir = True
else:
    base_clips_path = os.path.dirname(os.path.abspath(INPUT_VIDEO))
    used_custom_temp_dir = False


CLIPS_FOLDER = os.path.join(base_clips_path, "clips")


# -----------------------------------------
# Functions
# -----------------------------------------

def _run_cmd(cmd_list):
    """Run a command without shell=True, capture output, never raise."""
    return subprocess.run(cmd_list, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)

def find_and_extract_embedded_subtitle(video_path: str | Path, preferred_lang: str = "eng") -> str | None:
    """
    Look for a text-based subtitle stream inside the video (subrip/ass/ssa/mov_text/webvtt).
    Prefer preferred_lang when present. If found, convert to .srt in a temp dir and return it.
    """
    TEXT_CODECS = {"subrip", "ass", "ssa", "mov_text", "webvtt"}
    video_path = Path(video_path)
    if not video_path.exists():
        return None

    probe = _run_cmd([
        "ffprobe", "-v", "error",
        "-print_format", "json",
        "-show_streams",
        "-select_streams", "s",
        str(video_path)
    ])
    if probe.returncode != 0:
        return None

    try:
        data = json.loads(probe.stdout or "{}")
    except json.JSONDecodeError:
        return None

    streams = data.get("streams", [])
    candidates = []
    for s in streams:
        codec = (s.get("codec_name") or "").lower()
        if codec in {"subrip", "ass", "ssa", "mov_text", "webvtt"}:
            lang = (s.get("tags", {}).get("language") or "").lower()
            idx  = s.get("index")
            score = 2 if lang == (preferred_lang or "").lower() else 1
            candidates.append({"index": idx, "lang": lang or "und", "codec": codec, "score": score})

    if not candidates:
        return None

    candidates.sort(key=lambda c: (-c["score"], c["index"]))
    pick = candidates[0]

    tmpdir = Path(tempfile.mkdtemp(prefix="bleeparr_embeds_"))
    out_srt = tmpdir / f"{video_path.stem}.embedded.{pick['lang']}.srt"

    extract = _run_cmd([
        "ffmpeg", "-y", "-i", str(video_path),
        "-map", f"0:{pick['index']}",
        "-c:s", "srt",
        str(out_srt)
    ])
    if extract.returncode != 0 or not out_srt.exists() or out_srt.stat().st_size == 0:
        try: shutil.rmtree(tmpdir)
        except Exception: pass
        return None

    print(f"📦 Extracted embedded subtitle → {out_srt}")
    return str(out_srt)


def ensure_clips_folder():
    """Ensure the clips folder exists and is empty."""
    try:
        if os.path.exists(CLIPS_FOLDER):
            for f in os.listdir(CLIPS_FOLDER):
                os.remove(os.path.join(CLIPS_FOLDER, f))
        else:
            os.makedirs(CLIPS_FOLDER, exist_ok=True)
    except OSError as e:
        if getattr(e, "errno", None) == errno.ENOSPC:  # 28
            print(f"❌ Not enough space to create or clean clips folder:\n   {CLIPS_FOLDER}")
            print("👉 Free space on that filesystem OR rerun with a different location, e.g.:")
            print("   --temp-dir /tmp")
        else:
            print(f"❌ Failed to prepare clips folder: {CLIPS_FOLDER}")
            print(f"Error: {e}")
        exit(1)

        
def load_swears(swears_file):
    script_dir = os.path.dirname(os.path.realpath(__file__))
    swears_path = os.path.join(script_dir, swears_file)
    with open(swears_path, "r") as f:
        return set(line.strip() for line in f if line.strip())
                
def fuzzy_match(a: str, b: str, threshold: float = 0.8) -> bool:
    """
    Return True if strings a and b are similar enough.
    """
    return SequenceMatcher(None, a, b).ratio() >= threshold
    
def censor_word(word):
    """Return a censored version of a word like 'shit' -> 's***'."""
    if ALERT_CENSORING_OFF:
        # user asked to see the real word
        return word        
    if len(word) > 1:
        return word[0] + '*' * (len(word) - 1)
    else:
        return '*'


def find_or_download_subtitle(video_file, *, preferred_lang: str = "eng", allow_embedded: bool = True):
    """
    Resolution order:
      1) Existing external .srt or .hi.srt next to the video
      2) Embedded text subtitles inside the video (converted to .srt) [if allowed]
      3) Online search/download (Subliminal), retry once if empty
    Returns the path to an .srt or None if unavailable.
    """
    base = os.path.splitext(video_file)[0]
    possible_files = [base + ".srt", base + ".hi.srt"]

    # 1) Existing external SRT/HI-SRT
    for f in possible_files:
        if os.path.exists(f) and os.path.getsize(f) > 0:
            print(f"📄 Found existing subtitle: {f}")
            return f

    # 2) Embedded text subs → extract as SRT
    if allow_embedded:
        embedded = find_and_extract_embedded_subtitle(video_file, preferred_lang=preferred_lang)
        if embedded and os.path.exists(embedded) and os.path.getsize(embedded) > 0:
            return embedded

    # 3) Online search/download via Subliminal (2 tries at most)
    srt_file = base + ".srt"

    def download_subtitle_to_file():
        try:
            from subliminal import download_best_subtitles, region, Video
            from babelfish import Language
            region.configure('dogpile.cache.memory')
            video = Video.fromname(video_file)
            subtitles = download_best_subtitles([video], {Language(preferred_lang or 'eng')})
            if subtitles and video in subtitles and subtitles[video]:
                subtitle = subtitles[video][0]
                content = subtitle.content
                if content:
                    if isinstance(content, bytes):
                        try:
                            content = content.decode('utf-8')
                        except Exception:
                            content = content.decode('latin1', errors='ignore')
                    with open(srt_file, "w", encoding="utf-8") as f:
                        f.write(content)
                    return True
            return False
        except Exception as e:
            print(f"⚠️ Subliminal error: {e}")
            return False

    print(f"🌐 No local/embedded subtitle found. Attempting to download subtitles (lang={preferred_lang or 'eng'})...")
    success = download_subtitle_to_file()
    if not success or not os.path.exists(srt_file) or os.path.getsize(srt_file) == 0:
        print(f"⚠️ Subtitle file empty after first download attempt. Trying again...")
        try:
            if os.path.exists(srt_file):
                os.remove(srt_file)
        except Exception:
            pass
        success = download_subtitle_to_file()
        if not success or not os.path.exists(srt_file) or os.path.getsize(srt_file) == 0:
            print("❌ Subtitle download failed after two attempts. Cannot continue.")
            return None

    print(f"✅ Downloaded and saved subtitle: {srt_file}")
    return srt_file


def parse_subtitles(subtitle_file, swears):
    """Parse .srt file and return a list of bad sections."""
    with open(subtitle_file, "r", encoding="utf-8") as f:
        subtitles = list(srt.parse(f.read()))

    bad_sections = []
    for sub in subtitles:
        words = re.findall(r'\w+', sub.content.lower())
        bad_words = set(words) & swears
        if bad_words:
            bad_sections.append({
                "sub_index": sub.index,               # ← carry the .srt segment number
                "start":     sub.start.total_seconds(),
                "end":       sub.end.total_seconds(),
                "words":     list(bad_words),
            "content":   sub.content
            })
    return bad_sections

def extract_clips_segment(video_file, bad_sections, boost_db=6):
    """
    Use FFmpeg's segment muxer to split on every start/end time, then
    isolate exactly those subtitle‐defined segments:
      clip_01.wav, clip_02.wav, … clip_NN.wav.
    """
    
    # 1) clips folder already prepared by ensure_clips_folder()
    #next block is redundant:
    ## 1) wipe & recreate clips folder
    #if os.path.exists(CLIPS_FOLDER):
    #    shutil.rmtree(CLIPS_FOLDER)
    #os.makedirs(CLIPS_FOLDER)

    # 2) build split-points (every start and end time)
    times = []
    for sec in bad_sections:
        times.append(sec["start"])
        times.append(sec["end"])
    times = sorted(set(times))
    segment_times = ",".join(str(t) for t in times)

    # 3) one-shot FFmpeg split (mono, 16 kHz, zero-padded filenames)
    cmd = (
        f"ffmpeg -hide_banner -loglevel error -y -i \"{video_file}\" "
        f"-vn -acodec pcm_s16le -ac 1 -ar 16000 "
        f"-f segment -segment_times \"{segment_times}\" "
        f"-segment_start_number 1 \"{CLIPS_FOLDER}/clip_%02d.wav\""
    )
    
    print("🔧 extracting audio clips for each bad-word section…")
    subprocess.run(cmd, shell=True)

    # 4) isolate only the true segments (2nd,4th,6th,...)
    wanted = []
    for i in range(len(bad_sections)):
        seg_num = 2 * (i + 1)         # 2,4,6,...
        src_name = f"clip_{seg_num:02d}.wav"
        dst_name = f"clip_{i+1:02d}.wav"
        src_path = os.path.join(CLIPS_FOLDER, src_name)
        dst_path = os.path.join(CLIPS_FOLDER, dst_name)
        if os.path.exists(src_path):
            os.replace(src_path, dst_path)
            wanted.append(dst_name)
        else:
            print(f"⚠️ Warning: expected segment {src_name} not found")

    # 5) prune any leftovers (dummy, gaps, tail) from fresh listing
    # … your split & rename logic …
    for fname in os.listdir(CLIPS_FOLDER):
        if fname.endswith(".wav") and fname not in wanted:
            os.remove(os.path.join(CLIPS_FOLDER, fname))

    actual = len(wanted)

    # report exactly which subtitle each clip came from:
    for i, sec in enumerate(bad_sections, start=1):
        censored = ", ".join(censor_word(w) for w in sec["words"])
        print(f"🎯 Extracted clip_{i:02d}.wav from subtitle segment {sec['sub_index']}, [found bad words: {censored}]")

    print(f"✅ Extracted {actual} audio clips with bad words. Will now process the clips using Whisper AI")

    # 6) optional boost pass
    if boost_db > 0:
        for idx in range(1, actual + 1):
            clip_name = f"clip_{idx:02d}.wav"
            src = os.path.join(CLIPS_FOLDER, clip_name)
            boosted = src.replace(".wav", "_boosted.wav")
            boost_cmd = (
                f"ffmpeg -hide_banner -loglevel error -y "
                f"-i \"{src}\" -filter:a \"volume={boost_db}dB\" \"{boosted}\""
            )
            subprocess.run(boost_cmd, shell=True)
            os.replace(boosted, src)

def whisper_refine_clips_tiered_small(clips_folder, swears):
    """
    Run Whisper ‘small.en’ on every clip in `clips_folder`.
    Returns a tuple:
      (list_of_detected_hits, list_of_missed_clip_indices)
    where each hit is a dict with keys:
      clip_number, start, end, word, model
    """
    model = WhisperModel("small.en", device="cpu", compute_type="int8")
    hits = []
    missed = []
    clips = sorted(f for f in os.listdir(clips_folder) if f.endswith(".wav"))

    for idx, filename in enumerate(clips):
        path = os.path.join(clips_folder, filename)
        segments, _ = model.transcribe(path, beam_size=5, word_timestamps=True)

        found_this_clip = False
        for seg in segments:
            for wi in seg.words:
                w = wi.word.strip().lower()
                if w in swears:
                    hits.append({
                        "clip_number": idx,
                        "start": wi.start,
                        "end": wi.end,
                        "word": w,
                        "model": "small"
                    })
                    print(f"🔴 Found bad word '{censor_word(w)}' in {filename} [{wi.start:.2f}s → {wi.end:.2f}s] (small)")
                    found_this_clip = True
        if not found_this_clip:
            missed.append(idx)

    return hits, missed


def whisper_refine_clips_tiered_medium(clips_folder, swears, clip_indices):
    """
    Run Whisper ‘medium.en’ only on those clips whose indices
    are in clip_indices (i.e. that small missed).
    Returns (list_of_detected_hits, list_of_still_missed_indices).
    """
    model = WhisperModel("medium.en", device="cpu", compute_type="int8")
    hits = []
    still_missed = []

    clips = sorted(f for f in os.listdir(clips_folder) if f.endswith(".wav"))

    for idx in clip_indices:
        filename = clips[idx]
        path = os.path.join(clips_folder, filename)
        segments, _ = model.transcribe(path, beam_size=5, word_timestamps=True)

        found = False
        for seg in segments:
            for wi in seg.words:
                w = wi.word.strip().lower()
                if w in swears:
                    hits.append({
                        "clip_number": idx,
                        "start": wi.start,
                        "end": wi.end,
                        "word": w,
                        "model": "medium"
                    })
                    print(f"🔵 Found bad word '{censor_word(w)}' in {filename} [{wi.start:.2f}s → {wi.end:.2f}s] (medium)")
                    found = True
        if not found:
            still_missed.append(idx)

    return hits, still_missed

def merge_whisper_and_subtitles(whisper_bad_words, subtitle_bad_sections, fallback_enabled=True):
    """
    Merge Whisper hits with full‐subtitle fallback when ANY expected word is missing.
    Uses fuzzy matching so that, e.g., 'f***in' vs. 'f***ing' counts as a match.
    Returns (mute_segments, fallback_clip_indices).
    """
    mute_segments = []
    fallback_clips = []

    # 1) Build a map: clip_index -> list of whisper hits
    detected_map = defaultdict(list)
    for b in whisper_bad_words:
        detected_map[b["clip_number"]].append(b)

    # 2) Process each subtitle section in order
    for idx, sec in enumerate(subtitle_bad_sections):
        expected = set(sec["words"])
        heard    = [b["word"] for b in detected_map.get(idx, [])]

        # build the set of expected words that Whisper actually matched (exact or fuzzy)
        matched = set()
        for exp in expected:
            for h in heard:
                if h == exp or fuzzy_match(exp, h):
                    matched.add(exp)
                    break

        # 2a) Whisper found all expected words (exactly or fuzzily)?
        if matched == expected and expected:
            for b in detected_map[idx]:
                mute_segments.append({
                    "start":       b["start"],
                    "end":         b["end"],
                    "word":        b["word"],
                    "clip_number": idx,
                    "model":       b["model"]
                })

        # 2b) Partial or zero detection: fallback full subtitle mute
        else:
            if fallback_enabled:
                clip_num = idx + 1
                masked   = ", ".join(censor_word(w) for w in sec["words"])
                print(f"⚡ Full subtitle mute on clip_{clip_num:02d}.wav  →  [{masked}]")
                mute_segments.append({
                    "start":       sec["start"],
                    "end":         sec["end"],
                    "word":        sec["words"],
                    "clip_number": idx,
                    "fallback":    True
                })
                fallback_clips.append(idx)
            # else: skip completely if you only want full‐success clips

    # 3) Sort all segments by absolute start time
    mute_segments.sort(key=lambda x: x["start"])
    return mute_segments, fallback_clips

def build_and_run_ffmpeg(input_file, mute_points, pre_buffer_ms, post_buffer_ms, output_suffix):
    """Build and execute ffmpeg command to mute or beep bad sections."""
    output_file = os.path.splitext(input_file)[0] + f" {output_suffix}.mkv"

    if use_beep:
        filter_parts = []
        filter_parts.append("[0:a]anull[a0]")  # input labeled as a0
        current_label = "a0"

        # Step 1: apply mute filters (volume=0 during each range)
        for i, m in enumerate(mute_points):
            start = max(0, m["start"] - (pre_buffer_ms / 1000.0))
            end = m["end"] + (post_buffer_ms / 1000.0)
            next_label = f"muted{i}"
            filter_parts.append(
                f"[{current_label}]volume=enable='between(t,{start},{end})':volume=0[{next_label}]"
            )
            current_label = next_label

        # Step 2: create beeps at same times
        beep_labels = []
        for i, m in enumerate(mute_points):
            start = max(0, m["start"] - (pre_buffer_ms / 1000.0))
            end = m["end"] + (post_buffer_ms / 1000.0)
            duration = end - start
            delay_ms = int(start * 1000)

            beep = f"beep{i}"
            dbeep = f"dbeep{i}"

            filter_parts.append(f"aevalsrc=sin(2*PI*1000*t):d={duration}:s=44100[{beep}]")
            filter_parts.append(f"[{beep}]adelay={delay_ms}|{delay_ms}[{dbeep}]")
            beep_labels.append(f"[{dbeep}]")

        # Step 3: mix muted audio with all beep tracks
        all_inputs = f"[{current_label}]" + "".join(beep_labels)
        filter_parts.append(f"{all_inputs}amix=inputs={len(beep_labels)+1}:duration=longest[outa]")

        filter_complex = ";".join(filter_parts)
        cmd = (
            f"ffmpeg -hide_banner -loglevel error -y -i \"{input_file}\" "
            f"-filter_complex \"{filter_complex}\" -map 0:v -map \"[outa]\" "
            f"-c:v copy -c:a aac \"{output_file}\""
        )
    else:
        # Mute only (no beep)
        filters = []
        for m in mute_points:
            start = max(0, m["start"] - (pre_buffer_ms / 1000.0))
            end = m["end"] + (post_buffer_ms / 1000.0)
            filters.append(f"volume=enable='between(t,{start},{end})':volume=0")

        filter_str = ",".join(filters) if filters else "anull"
        cmd = (
            f"ffmpeg -hide_banner -loglevel error -y -i \"{input_file}\" "
            f"-af \"{filter_str}\" -c:v copy \"{output_file}\""
        )

    
    print(f"🔧 Running FFmpeg command to mute bad words{' with beeps' if use_beep else ''}…")
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    if result.returncode != 0:
        err = (result.stderr or "").lower()
        if "no space left on device" in err:
            print("❌ FFmpeg failed: No space left on device while writing output.")
            print("👉 Free space on the target drive OR rerun using a different temp location, e.g.:")
            print("   --temp-dir /tmp")
        else:
            print("❌ FFmpeg failed. Last lines from stderr:")
            print("\n".join(result.stderr.strip().splitlines()[-20:]))
        exit(1)
    
    return output_file


def cleanup_temp_clips(clips_folder):
    """Delete the entire clips folder and all contents after processing."""
    if os.path.exists(clips_folder):
        shutil.rmtree(clips_folder)
        print("🧹 Temp clips and folder cleaned up.")

# -----------------------------------------
# Main Run
# -----------------------------------------

if __name__ == "__main__":
    start_time = time.time()
    print("🚀 Starting Bleeparr...")

    swears = load_swears(SWEARS_FILE)

    if not INPUT_SRT:
        INPUT_SRT = find_or_download_subtitle(
            INPUT_VIDEO,
            preferred_lang=args.subtitle_lang,
            allow_embedded=not args.no_embedded_subs
        )
        if not INPUT_SRT:
            print("❌ No subtitles found (local/embedded/online). Exiting.")
            exit(1)
    

    bad_sections = parse_subtitles(INPUT_SRT, swears)
    
    if not bad_sections:
        print("✅ No bad words found in subtitles. Skipping Whisper and FFmpeg.")
    
        if DRY_RUN:
            print("🧪 Dry run: would have renamed the file and possibly deleted the subtitle. Exiting without changes.")
            exit(0)
    
        # Rename original video to match output style
        new_path = os.path.splitext(INPUT_VIDEO)[0] + f" {OUTPUT_SUFFIX}" + os.path.splitext(INPUT_VIDEO)[1]
        os.rename(INPUT_VIDEO, new_path)
        print(f"📁 Renamed input file to: {new_path}")
    
        # Remove subtitle if requested
        if not KEEP_SUBS and INPUT_SRT and os.path.exists(INPUT_SRT):
            os.remove(INPUT_SRT)
            print(f"🗑️ Deleted subtitle file: {INPUT_SRT}")
    
        print("\n🎉 Bleeparr finished successfully (no changes needed).")
        exit(0)
    
    if DRY_RUN:
        # Plan-only: use full subtitle segments as the conservative plan
        planned = [{
            "start": sec["start"],
            "end": sec["end"],
            "words": sec["words"]
        } for sec in bad_sections]
    
        print(f"🧪 Dry run: would process {len(planned)} subtitle segment(s).")
        for i, seg in enumerate(planned, 1):
            duration = seg["end"] - seg["start"]
            masked = ", ".join('*' * len(w) if not ALERT_CENSORING_OFF else w for w in seg["words"])
            print(f"  • Segment {i}: {seg['start']:.2f}s → {seg['end']:.2f}s (dur {duration:.2f}s)  [{masked}]")
    
        print("🧪 Dry run: would then run Whisper (small/medium as configured) and FFmpeg to apply mutes/beeps.")
        print("🧪 Dry run: no files were created/renamed/deleted.")
        exit(0)
    
    # only create clips folder when we actually need it (not in --dry-run)
    ensure_clips_folder()
    extract_clips_segment(INPUT_VIDEO, bad_sections, boost_db=BOOST_DB)

    # --- 1) Small Whisper pass over EVERY clip ---
    all_bad_words = []
    missed = list(range(len(bad_sections)))
    if DO_SMALL_MODEL:
        to_process = list(missed)
        small_hits, missed = whisper_refine_clips_tiered_small(CLIPS_FOLDER, swears)
        if small_hits:
            all_bad_words.extend(small_hits)
        else:
            print(f"ℹ️  Whisper Small pass ran on {len(to_process)} clips and found no bad words.")

    # --- 2) Medium Whisper pass on the clips small missed ---
    if DO_MEDIUM_MODEL and missed:
        # remember how many clips we’re checking
        to_process = list(missed)
        medium_hits, missed = whisper_refine_clips_tiered_medium(
            CLIPS_FOLDER, swears, to_process
        )
        if medium_hits:
            all_bad_words.extend(medium_hits)
        else:
            print(f"ℹ️  Whisper AI Medium.en ran on {len(to_process)} clips and found no additional bad words.")

    # --- 3) Full‐subtitle fallback for any clips still missed ---
    merged_mutes, fallback_clips = merge_whisper_and_subtitles(
        all_bad_words,
        bad_sections,
        fallback_enabled=DO_FULL_SUB_MUTE
    )

    # ── NEW: fix Whisper times to absolute video times ──
    for m in merged_mutes:
        if not m.get("fallback", False):
            base = bad_sections[m["clip_number"]]["start"]
            m["start"] = base + m["start"]
            m["end"]   = base + m["end"]

    # tally each stage
    small_hits   = sum(1 for b in all_bad_words if b["model"] == "small")
    medium_hits  = sum(1 for b in all_bad_words if b["model"] == "medium")
    fallback_hits = len(fallback_clips)

    print(f"\n📋 Mute Summary:")
    print(f"🔹 {small_hits} mutes detected by Whisper Small model")
    print(f"🔹 {medium_hits} mutes detected by Whisper Medium fallback")
    print(f"🔹 {fallback_hits} full subtitle fallback mutes")

    whisper_mutes = [m for m in merged_mutes if not m.get("fallback")]
    subtitle_mutes = [m for m in merged_mutes if m.get("fallback")]
    
    if beep_mode == "words":
        selected_mutes = whisper_mutes
    elif beep_mode == "segments":
        selected_mutes = subtitle_mutes
    elif beep_mode == "both":
        selected_mutes = merged_mutes
    else:
        selected_mutes = merged_mutes  # fallback if invalid
    
    print(f"🎛️ Beep mode = {beep_mode}. Using {len(selected_mutes)} segment(s).")
    
    #this line to use selected_mutes:
    output_file = build_and_run_ffmpeg(INPUT_VIDEO, selected_mutes, PRE_BUFFER_MS, POST_BUFFER_MS, OUTPUT_SUFFIX)

    if DELETE_ORIGINAL and 'output_file' in locals() and os.path.exists(output_file):
        if os.path.exists(INPUT_VIDEO):
            os.remove(INPUT_VIDEO)
            print(f"🗑️ Deleted original input file: {INPUT_VIDEO}")
        else:
            print(f"⚠️ Could not delete input — not found: {INPUT_VIDEO}")
            
    if not KEEP_SUBS and INPUT_SRT and os.path.exists(INPUT_SRT):
        os.remove(INPUT_SRT)
        print(f"🗑️ Deleted subtitle file: {INPUT_SRT}")

    # … your cleanup and other prints …

    if not retain_clips:
        cleanup_temp_clips(CLIPS_FOLDER)
        # Optional: remove --temp-dir folder if it was custom and is now empty
        if used_custom_temp_dir:
            try:
                if os.path.exists(base_clips_path) and not os.listdir(base_clips_path):
                    os.rmdir(base_clips_path)
                    print(f"🧹 Removed empty temp-dir folder: {base_clips_path}")
            except Exception as e:
                print(f"⚠️ Could not remove temp-dir folder {base_clips_path}: {e}")
    else:
        print(f"📦 Retaining clips folder as requested (--retain-clips set): {CLIPS_FOLDER}")

    elapsed = time.time() - start_time
    mins = int(elapsed // 60)
    secs = int(elapsed % 60)
    print(f"⏱️ Completed in {mins} min {secs} sec")

    print("\n🎉 Bleeparr finished successfully!")
