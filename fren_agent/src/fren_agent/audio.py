import os
import wave
import queue
import threading
from dataclasses import dataclass
from typing import Optional, Generator

import numpy as np
import sounddevice as sd
import webrtcvad
from faster_whisper import WhisperModel

@dataclass
class AudioConfig:
    sample_rate: int = 16000
    vad_frame_ms: int = 20
    vad_aggressiveness: int = 2
    stt_model_size: str = "base.en"
    push_to_talk: bool = True

class WhisperSTT:
    def __init__(self, model_size: str = "base.en", device: str = "cpu"):
        compute_type = "int8" if device == "cpu" else "float16"
        self.model = WhisperModel(model_size, device=(device if device == "cuda" else "cpu"),
                                  compute_type=compute_type)

    def transcribe_wav(self, path: str) -> str:
        segments, info = self.model.transcribe(path, beam_size=1)
        return " ".join(seg.text for seg in segments).strip()

class AudioRecorder:
    def __init__(self, cfg: AudioConfig):
        self.cfg = cfg
        self.stream = None
        self.frames_q = queue.Queue()
        self.vad = webrtcvad.Vad(self.cfg.vad_aggressiveness)
        self.recording = False
        self.channels = 1

    def _callback(self, indata, frames, time, status):
        if status:
            pass
        pcm16 = (indata[:, 0] * 32767).astype(np.int16).tobytes()
        self.frames_q.put(pcm16)

    def start(self):
        if self.recording:
            return
        self.recording = True
        self.stream = sd.InputStream(samplerate=self.cfg.sample_rate,
                                     channels=self.channels,
                                     dtype='float32',
                                     callback=self._callback)
        self.stream.start()

    def stop(self):
        if not self.recording:
            return
        self.recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def record_push_to_talk(self, seconds: float, out_wav: str) -> str:
        self.start()
        target_bytes = int(self.cfg.sample_rate * seconds * 2)  # 16-bit mono
        raw = b""
        while len(raw) < target_bytes:
            try:
                raw += self.frames_q.get(timeout=1.0)
            except queue.Empty:
                break
        self.stop()
        self._write_wav(out_wav, raw, self.cfg.sample_rate)
        return out_wav

    def stream_and_segment(self, out_dir: str, silence_ms: int = 600, max_len_ms: int = 15000) -> Generator[str, None, None]:
        """Yield wav paths for each VAD-detected utterance."""
        os.makedirs(out_dir, exist_ok=True)
        self.start()
        frame_bytes = int(self.cfg.sample_rate * (self.cfg.vad_frame_ms / 1000) * 2)
        buffer = b""
        voiced = False
        silence_count = 0
        max_frames = max_len_ms // self.cfg.vad_frame_ms
        cur_frames = 0
        idx = 0
        try:
            while self.recording:
                pcm = self.frames_q.get(timeout=1.0)
                buffer += pcm
                is_speech = self.vad.is_speech(pcm, self.cfg.sample_rate)
                cur_frames += 1
                if is_speech:
                    voiced = True
                    silence_count = 0
                elif voiced:
                    silence_count += 1
                if (voiced and silence_count * self.cfg.vad_frame_ms >= silence_ms) or cur_frames >= max_frames:
                    wav_path = os.path.join(out_dir, f"utt_{idx:04d}.wav")
                    self._write_wav(wav_path, buffer, self.cfg.sample_rate)
                    yield wav_path
                    idx += 1
                    buffer = b""
                    voiced = False
                    silence_count = 0
                    cur_frames = 0
        except queue.Empty:
            pass
        finally:
            self.stop()

    def _write_wav(self, path: str, pcm_bytes: bytes, sample_rate: int):
        with wave.open(path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_bytes)
