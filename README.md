# 🔇 Bleeparr_CLI 1.2

**Bleeparr_CLI** is an automated profanity censorship tool that intelligently detects and mutes (or beeps over) bad words in video files using subtitles and Whisper AI.

🚀 Built for creators, parents, teachers, and anyone who wants cleaner media.

---

## ✨ What's New in v1.2

- 🔊 `--beep`: Add a beep tone instead of muting bad words.
- 🎚️ `--beep-mode`: Choose between beeping just the detected words, the full subtitle segments, or both.
- 📁 `--temp-dir`: Choose a custom path for storing temporary clips.
- 📦 `--retain-clips`: Keep the `clips/` folder after processing (default: delete).

---

🆕 Changelog: Bleeparr v1.1

🔊 New Features
- Whisper Multi-Model Fallback: If Whisper Small misses a known bad word, it will retry using Whisper Medium before falling back >
- Flexible Censorship Display: New flag --alert-censoring-off allows disabling asterisk-style censoring for debugging/logging cla>
- More Accurate Clip Extraction: Extracted clips are now tightly trimmed to subtitle segment times, greatly reducing Whisper proc>
- Soft Matching for Swears: Fuzzy matching allows detection even when Whisper uses slight word variations (e.g., ****in vs ****in>
- Better Logging: Each clip now logs the subtitle segment number and which words were found there.
- Cleaner Output: Summary includes mute counts by detection method.

🧰 Improvements
- Boosted audio clip loudness by default (+6dB) for better Whisper accuracy.
- Improved CLI structure and argparse error handling.
- Supports subtitles with multiple bad words in a single line.
---


## ⚙️ Installation

Install system and Python dependencies:

```bash
sudo apt install ffmpeg
pip3 install -r requirements.txt
```

Python requirements include:

- `faster-whisper`
- `ffmpeg-python`
- `srt`
- `subliminal`
- `babelfish`

---

## 🎬 Basic Usage

```bash
python3 bleeparr.py --input yourvideo.mkv
```

This will:
- Auto-extract or download subtitles
- Detect profanity using Whisper + subtitles
- Mute the offending segments
- Output a remuxed `.mkv` with clean audio

---

## ✅ Command-Line Options

| Option                  | Description |
|-------------------------|-------------|
| `--input`               | **Required.** Path to input video file |
| `--subtitle`            | Optional subtitle file (.srt). Auto-detects if not provided |
| `--swears`              | File containing swear words (default: `swears.txt`) |
| `--boost-db`            | dB boost to improve Whisper accuracy (default: 6) |
| `--pre-buffer`          | Mute buffer (ms) before detected word (default: 100) |
| `--post-buffer`         | Mute buffer (ms) after detected word (default: 100) |
| `--output-suffix`       | Suffix for output file name (default: `(edited by Bleeparr)`) |
| `--delete-original`     | Delete the input video after success |
| `--no-keep-subs`        | Delete subtitle file after processing |
| `--alert-censoring-off` | Show uncensored words in terminal/log output |

### 🔊 Audio Control Options

| Option             | Description |
|--------------------|-------------|
| `--beep`           | Insert a tone instead of muting |
| `--beep-mode`      | `words` (default), `segments`, or `both` |
| `--bleeptool`      | Pass types to run: `S`, `M`, `FSM` (e.g. `S-M-FSM`) |

### 📁 Folder Options

| Option             | Description |
|--------------------|-------------|
| `--temp-dir`       | Custom directory for storing temporary audio clips |
| `--retain-clips`   | Retain the clips folder after processing |

---

## 📌 Examples

Basic mute (default):
```bash
python3 bleeparr.py --input movie.mkv
```

Use a custom swear list:
```bash
python3 bleeparr.py --input movie.mkv --swears mylist.txt
```

Beep instead of mute:
```bash
python3 bleeparr.py --input movie.mkv --beep
```

Beep full subtitle segments:
```bash
python3 bleeparr.py --input movie.mkv --beep --beep-mode segments
```

Custom clips folder:
```bash
python3 bleeparr.py --input movie.mkv --temp-dir /tmp/clips --retain-clips
```

---

## 🧠 How It Works

1. Subtitles are parsed to find likely profanity.
2. Whisper runs over extracted clips to verify words.
3. Final audio is remuxed using FFmpeg with volume filters or beeps.

Detection tiers:
- `S`: Whisper Small
- `M`: Whisper Medium (fallback)
- `FSM`: Full Subtitle Mute fallback

---

## ❤️ Credits

Huge thanks to [mmguero](https://github.com/mmguero) for inspiring this project with [cleanvid](https://github.com/mmguero/cleanvid) and [monkeyplug](https://github.com/mmguero/monkeyplug).

This tool is a personal coding journey — it's built with ❤️, powered by [ChatGPT](https://openai.com/chatgpt), and released freely.

---

## 📄 License

MIT License. Use it, fork it, improve it.
