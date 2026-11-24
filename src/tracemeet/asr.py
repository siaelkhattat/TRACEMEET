"""ASR helpers (faster-whisper)."""
import shutil
import time

import soundfile as sf


def pick_device():
    return "cuda" if shutil.which("nvidia-smi") else "cpu"


def load_model(name="small"):
    from faster_whisper import WhisperModel
    device = pick_device()
    compute = "float16" if device == "cuda" else "int8"
    print(f"loading faster-whisper '{name}' on {device} ({compute})")
    return WhisperModel(name, device=device, compute_type=compute)


def load_audio(wav_path):
    """Our data is already 16 kHz mono WAV: read it directly, skipping PyAV."""
    audio, sr = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        raise ValueError(f"{wav_path}: expected 16 kHz, got {sr}")
    return audio


def transcribe(model, wav_path, language="fr", beam_size=5):
    t0 = time.time()
    audio = load_audio(wav_path)
    segments, info = model.transcribe(
        audio, language=language, beam_size=beam_size,
        vad_filter=True, condition_on_previous_text=False)
    segs = [{"start": round(s.start, 2), "end": round(s.end, 2),
             "text": s.text.strip(),
             "avg_logprob": round(s.avg_logprob, 3),
             "no_speech_prob": round(s.no_speech_prob, 3)} for s in segments]
    return {"language": info.language, "duration": info.duration,
            "runtime_s": round(time.time() - t0, 1), "segments": segs}
