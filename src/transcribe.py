import os
import sys
import time
import subprocess
import numpy as np
from faster_whisper import WhisperModel

_models = {}

def load_model(size="small"):
    """Load each Whisper size once and reuse it."""
    if size not in _models:
        _models[size] = WhisperModel(size, device="cpu", compute_type="int8")
    return _models[size]

def load_audio(path, sr=16000):
    """Decode any audio/video file to a 16 kHz mono float32 array using ffmpeg."""
    cmd = ["ffmpeg", "-nostdin", "-i", path,
           "-f", "s16le", "-ac", "1", "-ar", str(sr), "-loglevel", "error", "-"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(out, np.int16).astype(np.float32) / 32768.0

def transcribe(audio_path, model_size="small", save_dir="data/transcripts"):
    """Transcribe an audio/video file. Returns (full_text, segments)."""
    model = load_model(model_size)
    audio = load_audio(audio_path)
    segments, info = model.transcribe(audio, beam_size=5, vad_filter=True)

    seg_list = []
    for seg in segments:
        seg_list.append({"start": seg.start, "end": seg.end, "text": seg.text.strip()})

    full_text = " ".join(s["text"] for s in seg_list)

    os.makedirs(save_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(audio_path))[0]

    with open(os.path.join(save_dir, f"{name}.txt"), "w", encoding="utf-8") as f:
        f.write(full_text)

    with open(os.path.join(save_dir, f"{name}_timestamps.txt"), "w", encoding="utf-8") as f:
        for s in seg_list:
            f.write(f"[{s['start']:.1f} - {s['end']:.1f}] {s['text']}\n")

    return full_text, seg_list

if __name__ == "__main__":
    path = sys.argv[1]
    size = sys.argv[2] if len(sys.argv) > 2 else "small"
    start = time.time()
    text, segs = transcribe(path, size)
    print(f"Transcription done in {time.time() - start:.1f}s")
    print(f"Segments: {len(segs)}, words: {len(text.split())}")
    print("\nPreview:\n", text[:500])