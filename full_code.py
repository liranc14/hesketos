#!/usr/bin/env python3
"""
Audio-to-Text transcription for הסכתוס podcast
- Downloads episodes from RSS
- Transcribes Hebrew audio using WhisperX
- Adds speaker diarization + timestamps (if available)
- Saves one TXT per episode
"""

import os
import sys
import traceback
import requests
import feedparser
import whisperx
from pyannote.audio import Pipeline

# ======================
# CONFIGURABLE PARAMETERS
# ======================
RSS_FEED_URL = "https://www.omnycontent.com/d/playlist/397b9456-4f75-4509-acff-ac0600b4a6a4/05f48c55-97c4-4049-8449-b14f00850082/e6bdb1ae-5412-42a1-a677-b14f008bbfc9/podcast.rss"
AUDIO_DIR = "audio_files"        # Where MP3s are saved
TEXT_DIR = "transcripts"         # Where transcripts are saved
DEVICE = "cpu"                   # "cpu" for t3.large (no GPU) cuda for GPU usage
MODEL_SIZE = "small"             # WhisperX model: "tiny", "base", "small", "medium", "large"
LANGUAGE = "he"                  # Hebrew transcription
USE_DIARIZATION = True           # Whether to perform speaker diarization
# ======================

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(TEXT_DIR, exist_ok=True)

# ======================
# Load models
# ======================
print(f"Loading WhisperX model ({MODEL_SIZE}, device={DEVICE})...")
whisper_model = whisperx.load_model(MODEL_SIZE, DEVICE, compute_type="float32")

diarization_pipeline = None
if USE_DIARIZATION:
    print("Loading speaker diarization pipeline...")
    try:
        diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token=None)
    except Exception as e:
        print("❌ Failed to load diarization pipeline. Speaker labels will be 'Unknown'.")
        traceback.print_exc()
        diarization_pipeline = None

# ======================
# Download podcast episodes
# ======================
print("Fetching RSS feed...")
feed = feedparser.parse(RSS_FEED_URL)

audio_files = []

for entry in feed.entries:
    if "enclosures" in entry and entry.enclosures:
        audio_url = entry.enclosures[0].href

        # Extract season & episode
        season = getattr(entry, "itunes_season", None)
        episode = getattr(entry, "itunes_episode", None)

        if season and episode:
            filename = f"{int(season)}{int(episode):02d}.mp3"
        elif episode:
            filename = f"{int(episode)}.mp3"
        else:
            filename = entry.title.replace("/", "_") + ".mp3"

        output_path = os.path.join(AUDIO_DIR, filename)

        if not os.path.exists(output_path):
            print(f"Downloading {filename}...")
            try:
                r = requests.get(audio_url, stream=True, timeout=30)
                r.raise_for_status()
                with open(output_path, "wb") as f:
                    for chunk in r.iter_content(1024 * 1024):
                        f.write(chunk)
            except Exception as e:
                print(f"❌ Failed to download {audio_url}:")
                traceback.print_exc()
                continue
        else:
            print(f"Already exists: {filename}")

        audio_files.append(output_path)

# ======================
# Transcribe and diarize
# ======================
filtered_audio_files = [f for f in audio_files if os.path.splitext(os.path.basename(f))[0].isdigit()]

for audio_path in filtered_audio_files:
    try:
        print(f"\nProcessing {audio_path} ...")

        # Transcription
        result = whisper_model.transcribe(audio_path, language=LANGUAGE)

        # Speaker diarization
        diarization = None
        if USE_DIARIZATION and diarization_pipeline:
            try:
                diarization = diarization_pipeline(audio_path)
            except Exception as e:
                print("❌ Diarization failed, using 'Unknown' for all speakers.")
                traceback.print_exc()

        # Merge transcription + speaker info
        text_output = ""
        for segment in result["segments"]:
            start = segment["start"]
            end = segment["end"]
            text = segment["text"]

            speaker = "Unknown"
            if diarization:
                for turn in diarization.itertracks(yield_label=True):
                    segment_start, segment_end, label = turn
                    if segment_end > start and segment_start < end:
                        speaker = label
                        break

            text_output += f"[{start:.2f}-{end:.2f}] {speaker}: {text}\n"

        # Save transcript
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        txt_path = os.path.join(TEXT_DIR, f"{base_name}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text_output)

        print(f"✅ Saved transcript: {txt_path}")

    except Exception as e:
        print(f"❌ Failed processing {audio_path}:")
        traceback.print_exc()
    finally:
        break

print("\nAll done!")
