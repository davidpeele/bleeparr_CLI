import os
import json
import subprocess
import argparse
import srt
import re
from faster_whisper import WhisperModel
from subliminal import download_best_subtitles, save_subtitles, region, Video
from babelfish import Language
import guessit

# -----------------------------------------
# CLI Parsing
# -----------------------------------------

parser = argparse.ArgumentParser(description="Bleeparr - Automatically censor videos based on subtitles and Whisper AI.")

parser.add_argument("--input", type=str, required=True, help="Input video file (e.g., 'testvideo.mkv')")
parser.add_argument("--subtitle", type=str, default=None, help="Optional: Subtitle file (e.g., 'testvideo.srt')")
parser.add_argument("--pre-buffer", type=int, default=100, help="Milliseconds to mute BEFORE bad word (default 100)")
parser.add_argument("--post-buffer", type=int, default=0, help="Milliseconds to mute AFTER bad word (default 0)")
parser.add_argument("--output-suffix", type=str, default=" (edited by Bleeparr)", help="Suffix for output file name")
parser.add_argument("--no-cleanup", action="store_true", help="Skip cleanup of temporary clips after processing")
parser.add_argument("--model", type=str, default="small.en", help="Whisper model size (default 'small.en')")
parser.add_argument("--boost-db", type=int, default=6, help="Audio boost in dB when extracting clips (default 6)")
parser.add_argument("--fallback-subtitle-mute", action="store_true", help="Mute full subtitle if Whisper misses a bad word")
parser.add_argument("--whisper-tiered", action="store_true", help="First pass with small.en, fallback to medium.en if needed")

args = parser.parse_args()

# -----------------------------------------
# Settings (from CLI)
# -----------------------------------------

INPUT_VIDEO = args.input
INPUT_SRT = args.subtitle
SWEARS_FILE = "swears.txt"
CLIPS_FOLDER = "clips"
WHISPER_MODEL_SIZE = args.model
PRE_BUFFER_MS = args.pre_buffer
POST_BUFFER_MS = args.post_buffer
OUTPUT_SUFFIX = args.output_suffix
DELETE_TEMP_CLIPS = not args.no_cleanup
BOOST_DB = args.boost_db
FALLBACK_SUBTITLE_MUTE = args.fallback_subtitle_mute
TIERED_WHISPER = args.whisper_tiered

# -----------------------------------------
# Functions
# -----------------------------------------

def ensure_clips_folder():
    if not os.path.exists(CLIPS_FOLDER):
        os.makedirs(CLIPS_FOLDER)

def load_swears(swears_file):
    with open(swears_file, "r", encoding="utf-8") as f:
        return set(word.strip().lower() for word in f if word.strip())

def find_or_download_subtitle(video_path):
    """Attempt to find embedded subtitles or download subtitles using subliminal."""

    base_name, _ = os.path.splitext(video_path)
    local_sub = f"{base_name}.srt"

    # Check if .srt file already exists
    if os.path.exists(local_sub):
        print(f"✅ Found existing subtitle file: {local_sub}")
        return local_sub

    # Step 1: Try to extract embedded subtitles
    print("🔍 Checking for embedded subtitles...")
    probe_cmd = ["ffprobe", "-v", "error", "-select_streams", "s:s", "-show_entries", "stream=index", "-of", "csv=p=0", video_path]
    result = subprocess.run(probe_cmd, capture_output=True, text=True)
    subtitle_streams = result.stdout.strip().split("\n") if result.stdout.strip() else []

    if subtitle_streams and subtitle_streams[0]:
        print(f"🎯 Embedded subtitles found! Extracting...")
        stream_index = subtitle_streams[0]
        extract_cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-map", f"0:s:{stream_index}",
            local_sub
        ]
        subprocess.run(extract_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if os.path.exists(local_sub):
            print(f"✅ Successfully extracted subtitles to {local_sub}")
            return local_sub

    # Step 2: Try downloading using subliminal
    print("🌐 No embedded subs found. Trying to download English subtitles...")

    try:
        region.configure('dogpile.cache.memory')

        video_metadata = Video.fromname(os.path.basename(video_path))
        subtitles = download_best_subtitles(
            [video_metadata],
            {Language('eng')},
            hearing_impaired=False
        )

        if subtitles.get(video_metadata):
            save_subtitles(video_metadata, subtitles[video_metadata])

            # Fix file name if needed
            downloaded_srt = next((f for f in os.listdir('.') if f.endswith('.srt') and base_name in f), None)
            if downloaded_srt and downloaded_srt != local_sub:
                print(f"📦 Renaming downloaded subtitle {downloaded_srt} -> {local_sub}")
                os.rename(downloaded_srt, local_sub)

            print(f"✅ Successfully downloaded and prepared subtitles for {video_path}")
            return local_sub
        else:
            print("❌ No English subtitles found via subliminal.")
            return None
    except Exception as e:
        print(f"❌ Error during subtitle download: {e}")
        return None

import re

def parse_subtitles(subtitle_file, swears):
    with open(subtitle_file, "r", encoding="utf-8") as f:
        content = f.read()
    subtitles = list(srt.parse(content))

    bad_sections = []
    for idx, sub in enumerate(subtitles):
        text = sub.content.lower()
        words = re.findall(r'\b\w+\b', text)  # split into words using word boundaries
        if any(word in swears for word in words):
            bad_sections.append({
                "clip_number": idx,
                "start": sub.start.total_seconds(),
                "end": sub.end.total_seconds(),
                "text": sub.content.strip()
            })
    return bad_sections

def extract_clips(input_video, bad_sections, boost_db=6):
    for i, section in enumerate(bad_sections):
        output_clip = os.path.join(CLIPS_FOLDER, f"clip_{i}.wav")
        start = section["start"]
        duration = section["end"] - section["start"]
        cmd = [
            "ffmpeg", "-y", "-i", input_video,
            "-ss", str(start),
            "-t", str(duration),
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            "-af", f"volume={boost_db}dB",  # 💥 Dynamic Boost
            output_clip
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"🎯 Extracted and boosted {boost_db}dB: {output_clip}")

def whisper_refine_clips(clips_folder, swears, model_size="small.en"):
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    all_bad_words = []
    clips = sorted(os.listdir(clips_folder))

    for i, clip_filename in enumerate(clips):
        if not clip_filename.endswith(".wav"):
            continue
        clip_path = os.path.join(clips_folder, clip_filename)
        segments, _ = model.transcribe(clip_path, beam_size=5, word_timestamps=True)

        for segment in segments:
            for word_info in segment.words:
                word = word_info.word.strip().lower()
                if word in swears:
                    start_time = word_info.start
                    end_time = word_info.end
                    print(f"🔴 Found bad word '{word}' in {clip_filename} from {start_time:.2f}s to {end_time:.2f}s")
                    all_bad_words.append({
                        "clip_number": i,
                        "start": start_time,
                        "end": end_time,
                        "word": word
                    })

    return all_bad_words

def shift_bad_words(all_bad_words, bad_sections):
    shifted = []
    for word_info in all_bad_words:
        clip_num = word_info["clip_number"]
        clip_start = bad_sections[clip_num]["start"]
        shifted.append({
            "start": max(clip_start + word_info["start"], 0),
            "end": clip_start + word_info["end"],
            "word": word_info["word"]
        })
    return shifted

def build_and_run_ffmpeg(input_video, shifted_bad_words, pre_buffer_ms=100, post_buffer_ms=0, output_suffix=" (edited by Bleeparr)"):
    pre_buffer_sec = pre_buffer_ms / 1000
    post_buffer_sec = post_buffer_ms / 1000

    filters = []
    for word in shifted_bad_words:
        start = max(word["start"] - pre_buffer_sec, 0)
        end = word["end"] + post_buffer_sec
        filters.append(f"volume=enable='between(t,{start},{end})':volume=0")

    audio_filter = ",".join(filters)

    base_name, ext = os.path.splitext(input_video)
    output_video = f"{base_name}{output_suffix}{ext}"

    ffmpeg_cmd = [
        "ffmpeg", "-i", input_video,
        "-af", audio_filter,
        "-c:v", "copy", "-y",
        output_video
    ]

    print(f"🔧 Running FFmpeg command:\n{' '.join(ffmpeg_cmd)}")
    subprocess.run(ffmpeg_cmd)

    print(f"\n✅ Finished! Output saved to: {output_video}")

def cleanup_temp_clips(clips_folder):
    for f in os.listdir(clips_folder):
        if f.endswith(".wav"):
            os.remove(os.path.join(clips_folder, f))
    print("🧹 Temp clips cleaned up.")


def merge_whisper_and_subtitles(all_bad_words, bad_sections, fallback_enabled):
    """Combine Whisper detections and subtitle fallback if needed."""
    merged_mutes = []

    # Organize whisper results by clip number
    whisper_by_clip = {}
    for word_info in all_bad_words:
        whisper_by_clip.setdefault(word_info["clip_number"], []).append(word_info)

    for idx, section in enumerate(bad_sections):
        if idx in whisper_by_clip:
            # ✅ Whisper found something for this clip, use fine-grained mute
            for word_info in whisper_by_clip[idx]:
                merged_mutes.append({
                    "start": max(section["start"] + word_info["start"], 0),
                    "end": section["start"] + word_info["end"],
                    "word": word_info["word"]
                })
        else:
            # ❌ Whisper missed it. If fallback enabled, mute the whole subtitle segment
            if fallback_enabled:
                print(f"⚡ Fallback mute for subtitle {idx} '{section['text']}'")
                merged_mutes.append({
                    "start": section["start"],
                    "end": section["end"],
                    "word": "[fallback mute]"
                })

    return merged_mutes


def whisper_refine_clips_tiered(clips_folder, swears):
    """Tiered whisper: try small.en first, fallback to medium.en on misses."""
    from faster_whisper import WhisperModel

    small_model = WhisperModel("small.en", device="cpu", compute_type="int8")
    all_bad_words = []
    missed_clips = []

    clips = sorted(os.listdir(clips_folder))

    for i, clip_filename in enumerate(clips):
        if not clip_filename.endswith(".wav"):
            continue
        clip_path = os.path.join(clips_folder, clip_filename)
        segments, _ = small_model.transcribe(clip_path, beam_size=5, word_timestamps=True)

        found = False
        for segment in segments:
            for word_info in segment.words:
                word = word_info.word.strip().lower()
                if word in swears:
                    start_time = word_info.start
                    end_time = word_info.end
                    print(f"🔴 Found bad word '{word}' in {clip_filename} [small.en]")
                    all_bad_words.append({
                        "clip_number": i,
                        "start": start_time,
                        "end": end_time,
                        "word": word
                    })
                    found = True

        if not found:
            missed_clips.append(i)

    # Step 2: Retry missed clips with medium.en
    if missed_clips:
        print(f"\n⚡ Retrying {len(missed_clips)} clips using medium.en model...")

        # 💥 FREE MEMORY FROM small.en
        del small_model
        import gc
        gc.collect()

        medium_model = WhisperModel("medium.en", device="cpu", compute_type="int8")

        for i in missed_clips:
            clip_filename = clips[i]
            clip_path = os.path.join(clips_folder, clip_filename)
            segments, _ = medium_model.transcribe(clip_path, beam_size=5, word_timestamps=True)

            for segment in segments:
                for word_info in segment.words:
                    word = word_info.word.strip().lower()
                    if word in swears:
                        start_time = word_info.start
                        end_time = word_info.end
                        print(f"🔵 Found bad word '{word}' in {clip_filename} [medium.en]")
                        all_bad_words.append({
                            "clip_number": i,
                            "start": start_time,
                            "end": end_time,
                            "word": word
                        })

    return all_bad_words




# -----------------------------------------
# Main Run
# -----------------------------------------

if __name__ == "__main__":
    print("🚀 Starting Bleeparr...")

    ensure_clips_folder()
    swears = load_swears(SWEARS_FILE)

    # Auto-handle missing subtitles
    if not INPUT_SRT:
        INPUT_SRT = find_or_download_subtitle(INPUT_VIDEO)
        if not INPUT_SRT:
            print("❌ No subtitles found or could not download. Exiting.")
            exit(1)

    bad_sections = parse_subtitles(INPUT_SRT, swears)
    extract_clips(INPUT_VIDEO, bad_sections, boost_db=BOOST_DB)
    if TIERED_WHISPER:
        all_bad_words = whisper_refine_clips_tiered(CLIPS_FOLDER, swears)
    else:
        all_bad_words = whisper_refine_clips(CLIPS_FOLDER, swears, model_size=WHISPER_MODEL_SIZE)
    merged_mutes = merge_whisper_and_subtitles(all_bad_words, bad_sections, fallback_enabled=FALLBACK_SUBTITLE_MUTE)

    # Print a clean summary:
    whisper_mutes = sum(1 for m in merged_mutes if m["word"] != "[fallback mute]")
    fallback_mutes = sum(1 for m in merged_mutes if m["word"] == "[fallback mute]")

    print(f"\n📋 Mute Summary:")
    print(f"🔹 {whisper_mutes} mutes detected by Whisper AI")
    print(f"🔹 {fallback_mutes} fallbacks to full subtitle muting")

    build_and_run_ffmpeg(INPUT_VIDEO, merged_mutes, pre_buffer_ms=PRE_BUFFER_MS, post_buffer_ms=POST_BUFFER_MS, output_suffix=OUTPUT_SUFFIX)


    if DELETE_TEMP_CLIPS:
        cleanup_temp_clips(CLIPS_FOLDER)

    print("\n🎉 Bleeparr finished successfully!")
