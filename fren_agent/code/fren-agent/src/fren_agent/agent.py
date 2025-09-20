from __future__ import annotations

# NOTE: keep this module lightweight at import time.
# Do NOT import cv2 / transformers / torch hubs here.
# We only declare config dataclass + shell class, and import heavy deps inside __init__.

from dataclasses import dataclass
from typing import Optional, List, Dict, Any

@dataclass
class AgentConfig:
    device: str
    blip_model: str
    text_model: str
    max_new_tokens: int
    temperature: float
    top_p: float
    sigma: float
    tau: float
    alpha: float
    primes: int
    audio_sample_rate: int
    vad_frame_ms: int
    vad_aggressiveness: int
    stt_model_size: str
    push_to_talk: bool
    tts_engine: str
    piper_bin: str
    piper_model: str
    piper_out: str
    mem_emb: str
    mem_index: str
    mem_meta: str
    mem_k: int
    cam_index: int
    fps: int

class FrenAgent:
    def __init__(self, cfg: AgentConfig, governance: Dict[str, Any] = None):
        from .vision import VisionCaptioner
        from .emotion import EmotionAnalyzer
        from .memory import SemanticMemory
        from .audio import AudioConfig, AudioRecorder, WhisperSTT
        from .tts import PiperTTS, FallbackTTS
        from .llm.resonant_llm import ResonantResponder
        import cv2
        import os

        self.cfg = cfg

        self.vision = VisionCaptioner(cfg.blip_model, cfg.device)
        self.emotion = EmotionAnalyzer()
        self.memory = SemanticMemory(cfg.mem_emb, cfg.mem_index, cfg.mem_meta, governance=governance)

        self.cfg = cfg

        # Subsystems
        self.vision = VisionCaptioner(cfg.blip_model, cfg.device)
        self.emotion = EmotionAnalyzer()
        self.memory = SemanticMemory(cfg.mem_emb, cfg.mem_index, cfg.mem_meta)
        self.asr = WhisperSTT(cfg.stt_model_size, cfg.device)
        self.rec = AudioRecorder(
            AudioConfig(
                sample_rate=cfg.audio_sample_rate,
                vad_frame_ms=cfg.vad_frame_ms,
                vad_aggressiveness=cfg.vad_aggressiveness,
                stt_model_size=cfg.stt_model_size,
                push_to_talk=cfg.push_to_talk
            ),
            device=cfg.device
        )

        try:
            if cfg.tts_engine == "piper":
                self.tts = PiperTTS(cfg.piper_bin, cfg.piper_model, cfg.piper_out)
            else:
                self.tts = FallbackTTS()
        except Exception:
            # fallback if Piper misconfigured
            self.tts = FallbackTTS()

        self.llm = ResonantResponder(
            model_name=cfg.text_model,
            device=cfg.device,
            sigma=cfg.sigma,
            tau=cfg.tau,
            alpha=cfg.alpha,
            primes=cfg.primes
        )

        # Camera (open late)
        self.cv2 = cv2
        self.cap = cv2.VideoCapture(cfg.cam_index, cv2.CAP_DSHOW if os.name == "nt" else 0)
        self.cap.set(cv2.CAP_PROP_FPS, cfg.fps)

        self._last_caption: str = ""

    # -------- Orchestration methods (same as before) --------
    def read_frame_and_caption(self) -> str:
        ok, frame = self.cap.read()
        if not ok or frame is None:
            return ""
        caption = self.vision.caption_frame(frame)
        self._last_caption = caption
        return caption

    def get_last_caption(self) -> str:
        return self._last_caption

    def stt_once(self, seconds: float = 4.0, tmp_wav: str = "./assets/utt.wav") -> str:
        path = self.rec.record_push_to_talk(seconds, tmp_wav)
        return self.asr.transcribe_wav(path)

    def stt_stream_iter(self, out_dir: str = "./assets/utt_stream"):
        yield from self.rec.stream_and_segment(out_dir)

    def respond(self, user_text: str) -> str:
        recalls = self.memory.search(user_text, k=self.cfg.mem_k)
        recall_texts = [r["text"] for r in recalls]

        caption = self.get_last_caption()
        sys_ctx = ""
        if caption:
            sys_ctx += f"[VISION] Scene: {caption}\n"
        sys_ctx += "[TASK] You are Fren-Agent, concise, helpful, technical when needed.\n"
        prompt = sys_ctx + f"\n[USER] {user_text}\n[ASSISTANT]"

        out, _diag = self.llm.generate(
            prompt=prompt,
            max_new_tokens=self.cfg.max_new_tokens,
            temperature=self.cfg.temperature,
            top_p=self.cfg.top_p,
            memory_snippets=recall_texts
        )

        if user_text.strip():
            self.memory.add(user_text, {"source": "user"})
        if out.strip():
            self.memory.add(out, {"source": "assistant"})
        return out

    def say(self, text: str):
        try:
            self.tts.speak(text)
        except Exception as e:
            print(f"[tts] error: {e}")

    def release(self):
        try:
            if getattr(self, "cap", None):
                self.cap.release()
        except Exception:
            pass
