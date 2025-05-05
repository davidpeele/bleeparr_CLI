# Bleeparr_CLI 1.1


🚀  **Bleeparr_CLI** is a tool for automatically detecting and censoring profanity in videos by muting bad words based on subtitles and AI speech recognition (Whisper).

It uses a smart multi-layered strategy:
- Subtitle scanning for obvious profanity
- Whisper AI transcription (small + medium fallback) for audio verification
- Final fallback to mute entire subtitle segments if needed
- Fast, safe, accurate censoring!

Big thanks to mmguero for his awesome (and better) tools that gave me the ideas for this:
- https://github.com/mmguero/cleanvid
- https://github.com/mmguero/monkeyplug

🆕 Changelog: Bleeparr v1.1

🔊 New Features
- Whisper Multi-Model Fallback: If Whisper Small misses a known bad word, it will retry using Whisper Medium before falling back to subtitle-based muting.
- Flexible Censorship Display: New flag --alert-censoring-off allows disabling asterisk-style censoring for debugging/logging clarity.
- More Accurate Clip Extraction: Extracted clips are now tightly trimmed to subtitle segment times, greatly reducing Whisper processing time.
- Soft Matching for Swears: Fuzzy matching allows detection even when Whisper uses slight word variations (e.g., ****in vs ****ing).
- Better Logging: Each clip now logs the subtitle segment number and which words were found there.
- Cleaner Output: Summary includes mute counts by detection method.

🧰 Improvements
- Boosted audio clip loudness by default (+6dB) for better Whisper accuracy.
- Improved CLI structure and argparse error handling.
- Supports subtitles with multiple bad words in a single line.
---

## 🛠 Installation

1. Install system dependencies:
   ```bash
   sudo apt install ffmpeg

2. Install Python dependencies:
   ```bash
   pip3 install -r requirements.txt

---

## ⚙️ How to Use

   ```bash
   python3 bleeparr.py --input yourvideo.mkv
   ```


✅ Common Command Line Options (CLI):

### Command-Line Options

| Option                 | Description                                                  | Default                    |
|------------------------|--------------------------------------------------------------|----------------------------|
| `--input`              | **Required.** Input video file path                          | None                       |
| `--subtitle`           | Optional subtitle file path (.srt)                           | Auto-detect/download       |
| `--swears`             | File of swear words to censor                                | `swears.txt`               |
| `--boost-db`           | Audio boost level (in dB) for extracted clips                | `6`                        |
| `--pre-buffer`         | Pre-mute buffer in milliseconds                              | `100`                      |
| `--post-buffer`        | Post-mute buffer in milliseconds                             | `100`                      |
| `--bleeptool`          | Passes to run: `S`, `M`, `FSM`, or combo (e.g., `S-M-FSM`)   | `S-M-FSM`                  |
| `--output-suffix`      | Suffix to append to final output filename                   | *(edited by Bleeparr)*     |
| `--delete-original`    | Delete input file after successful processing                | Off                        |
| `--no-keep-subs`       | Delete subtitle file after processing                        | Off                        |
| `--alert-censoring-off`| Print full swear words in logs instead of masking them       | Off                        |

Example command:
```
# Basic use with automatic subtitle detection and default options
python3 bleeparr.py --input my_video.mkv

# Use a custom list of swear words
python3 bleeparr.py --input my_video.mkv --swears custom_swears.txt

# Boost audio by 10 dB and disable subtitle cleanup
python3 bleeparr.py --input my_video.mkv --boost-db 10 --no-keep-subs

# Run only Whisper Small and fallback subtitle mute
python3 bleeparr.py --input my_video.mkv --bleeptool S-FSM

# Disable alert masking to show full swear words in terminal
python3 bleeparr.py --input my_video.mkv --alert-censoring-off

# Run and delete the original file afterward
python3 bleeparr.py --input my_video.mkv --delete-original```
---
```


🧠 Advanced Notes

	•	Subtitles required: Bleeparr will try to extract embedded subtitles, or download them automatically using Subliminal.
	•	Audio formats: Should work with most common formats (AAC, EAC3, Vorbis, etc.)
	•	Final Output: Video is remuxed with muted audio at swear locations.

⸻

❤️ Thanks (This is my first ever attempt at developing code. Full disclosure, it's pretty much 100% written by ChatGPT)

Bleeparr uses:

	•	Faster-Whisper
	•	Subliminal
	•	FFmpeg
 	•	and some other stuff...
  
