#!/usr/bin/env python3
"""
Audio-to-Text transcription for הסכתוס podcast
- Downloads episodes from RSS
- Transcribes Hebrew audio using WhisperX
- Adds speaker diarization + timestamps (if available)
- Saves one TXT per episode
- Supports batching by minutes and a test mode for quick testing
"""

import os
import sys
import traceback
import requests
import feedparser
import whisperx
from pyannote.audio import Pipeline
import torchaudio

# ======================
# CONFIGURABLE PARAMETERS
# ======================
RSS_FEED_URL = "https://www.omnycontent.com/d/playlist/397b9456-4f75-4509-acff-ac0600b4a6a4/05f48c55-97c4-4049-8449-b14f00850082/e6bdb1ae-5412-42a1-a677-b14f008bbfc9/podcast.rss"
AUDIO_DIR = "audio_files"        # Where MP3s are saved
TEXT_DIR = "transcripts"         # Where transcripts are saved
DEVICE = "cuda"                   # "cpu" for t3.large (no GPU) cuda for GPU usage
MODEL_SIZE = "small"             # WhisperX model: "tiny", "base", "small", "medium", "large"
LANGUAGE = "he"                  # Hebrew transcription
USE_DIARIZATION = True           # Whether to perform speaker diarization

# New parameters
BATCH_MINUTES = 2                # None for full file, or number of minutes per batch
TEST_FIRST = True                # True to process only first batch and exit
# ======================


# List of slang words
slang_words = ["גומי גומיהו", "להיטוס", "כישופון", "איכסי פיכסי"]


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

# Filter to only numeric filenames
filtered_audio_files = [f for f in audio_files if os.path.splitext(os.path.basename(f))[0].isdigit()]

# ======================
# Transcribe and diarize
# ======================
for audio_path in filtered_audio_files:
    try:
        print(f"\nProcessing {audio_path} ...")

        # Load full waveform
        waveform, sr = torchaudio.load(audio_path)

        # Compute batch size in samples
        if BATCH_MINUTES is not None:
            batch_samples = int(BATCH_MINUTES * 60 * sr)
        else:
            batch_samples = waveform.shape[1]

        # Split waveform into batches
        batches = []
        start = 0
        while start < waveform.shape[1]:
            end = min(start + batch_samples, waveform.shape[1])
            batches.append(waveform[:, start:end])
            start = end

        # Quick test mode: only first batch
        if TEST_FIRST and batches:
            batches = [batches[0]]

        # Store all segments across batches
        all_segments = []
        for i, batch_waveform in enumerate(batches):
            batch_numpy = batch_waveform.numpy()
            result = whisper_model.transcribe(batch_numpy, language=LANGUAGE, batch_size=1)

            offset_sec = i * (BATCH_MINUTES * 60 if BATCH_MINUTES else 0)
            for seg in result["segments"]:
                seg["start"] += offset_sec
                seg["end"] += offset_sec
                all_segments.append(seg)

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
        for segment in all_segments:
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

        # Exit if test mode
        if TEST_FIRST:
            print("\nTest mode enabled: stopping after first batch.")
            break

    except Exception as e:
        print(f"❌ Failed processing {audio_path}:")
        traceback.print_exc()
    finally:
        if TEST_FIRST:
            break

print("\nAll done!")
