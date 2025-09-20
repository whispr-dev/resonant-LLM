import os
import io
import wave
import queue
from dataclasses import dataclass
from typing import Optional, Generator

import numpy as np
import sounddevice as sd
import torch
from faster_whisper import WhisperModel

@dataclass
class AudioConfig:
    sample_rate: int = 16000
    vad_frame_ms: int = 20          # still used for chunk pacing
    vad_aggressiveness: int = 2     # unused here, kept for cfg compatibility
    stt_model_size: str = "base.en"
    push_to_talk: bool = True

def _load_silero_vad(device: str = "cpu"):
    dev = "cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu"
    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
        onnx=False
    )
    (get_speech_ts, _, read_audio, _, collect_chunks) = utils
    model.to(dev).eval()
    return model, get_speech_ts, collect_chunks, dev

class WhisperSTT:
    def __init__(self, model_size: str = "base.en", device: str = "cpu"):
        compute_type = "int8" if device != "cuda" else "float16"
        self.model = WhisperModel(model_size, device=("cuda" if device == "cuda" else "cpu"),
                                  compute_type=compute_type)

    def transcribe_wav(self, path: str) -> str:
        segments, _ = self.model.transcribe(path, beam_size=1)
        return " ".join(seg.text for seg in segments).strip()

class AudioRecorder:
    """
    Uses sounddevice for capture and Silero-VAD (Torch) for speech detection.
    No native extensions required on Windows.
    """
    def __init__(self, cfg: AudioConfig, device: str = "cpu"):
        self.cfg = cfg
        self.frames_q: "queue.Queue[bytes]" = queue.Queue()
        self.stream = None
        self.recording = False
        self.channels = 1
        self.model, self.get_speech_ts, self.collect_chunks, self.vad_device = _load_silero_vad(device)

    def _callback(self, indata, frames, time, status):
        if status:
            pass
        pcm16 = (indata[:, 0] * 32767.0).astype(np.int16).tobytes()
        self.frames_q.put(pcm16)

    def start(self):
        if self.recording:
            return
        self.recording = True
        self.stream = sd.InputStream(
            samplerate=self.cfg.sample_rate,
            channels=self.channels,
            dtype='float32',
            callback=self._callback
        )
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
        need_bytes = int(self.cfg.sample_rate * seconds * 2)  # 16-bit mono
        raw = b""
        while len(raw) < need_bytes:
            try:
                raw += self.frames_q.get(timeout=1.0)
            except queue.Empty:
                break
        self.stop()
        self._write_wav(out_wav, raw, self.cfg.sample_rate)
        return out_wav

    def stream_and_segment(self, out_dir: str, silence_ms: int = 600, max_len_ms: int = 15000) -> Generator[str, None, None]:
        """
        Continuous capture; use Silero VAD over a rolling buffer.
        Yields utterance wav paths when speech ends (by gap or max_len).
        """
        os.makedirs(out_dir, exist_ok=True)
        self.start()

        sr = self.cfg.sample_rate
        gap_samples = int((silence_ms/1000.0) * sr)
        max_samples = int((max_len_ms/1000.0) * sr)

        rolling = np.zeros(0, dtype=np.int16)
        voiced_active = False
        last_speech_end = 0
        idx = 0

        try:
            while self.recording:
                try:
                    chunk = self.frames_q.get(timeout=1.0)
                except queue.Empty:
                    continue
                # append to rolling buffer
                new = np.frombuffer(chunk, dtype=np.int16)
                rolling = np.concatenate([rolling, new])

                # run VAD in 0.5s windows for speed
                if len(rolling) < sr // 2:
                    continue

                # Silero expects float32 [-1..1]
                audio_float32 = (rolling.astype(np.float32) / 32768.0)
                tensor = torch.from_numpy(audio_float32).to(self.vad_device)
                with torch.no_grad():
                    speech_ts = self.get_speech_ts(tensor, self.model, sampling_rate=sr)

                if speech_ts:
                    voiced_active = True
                    # end = last speech end sample
                    last_speech_end = max([seg["end"] for seg in speech_ts])
                else:
                    # no speech currently detected
                    if voiced_active and (len(rolling) - last_speech_end >= gap_samples or len(rolling) >= max_samples):
                        # cut utterance at last_speech_end
                        cut = rolling[:last_speech_end] if last_speech_end > 0 else rolling
                        wav_path = os.path.join(out_dir, f"utt_{idx:04d}.wav")
                        self._write_wav(wav_path, cut.tobytes(), sr)
                        yield wav_path
                        idx += 1
                        # keep tail after last_speech_end to seed the next window
                        rolling = rolling[last_speech_end:] if last_speech_end > 0 else np.zeros(0, dtype=np.int16)
                        voiced_active = False
                        last_speech_end = 0
                    else:
                        # trim overly long buffers anyway
                        if len(rolling) > max_samples * 2:
                            rolling = rolling[-max_samples:]
        finally:
            self.stop()

    def _write_wav(self, path: str, pcm_bytes: bytes, sample_rate: int):
        with wave.open(path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_bytes)
