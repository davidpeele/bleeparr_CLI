# Bleeparr


🚀  **Bleeparr** is a tool for automatically detecting and censoring profanity in videos by muting bad words based on subtitles and AI speech recognition (Whisper).

It uses a smart multi-layered strategy:
- Subtitle scanning for obvious profanity
- Whisper AI transcription (small + medium fallback) for audio verification
- Final fallback to mute entire subtitle segments if needed
- Fast, safe, accurate censoring!

Big thanks to mmguero for his awesome (and better) tools that gave me the ideas for this:
- https://github.com/mmguero/cleanvid
- https://github.com/mmguero/monkeyplug

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


✅ Common options:


	•	--boost-db 6 (default) — Boost clip audio volume before analyzing
 
	•	--fallback-subtitle-mute — Mute the entire subtitle line if Whisper misses a swear
 
	•	--whisper-tiered — First use Whisper small.en for speed, then retry misses with medium.en
 
	•	--output-suffix "(edited by Bleeparr)" — Change output filename
 
	•	--model small.en or --model medium.en — Set the Whisper model (if not using tiered mode)

Example command:
```
python3 bleeparr.py --input "testvideo.mkv" --boost-db 6 --fallback-subtitle-mute --whisper-tiered
```
---



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
