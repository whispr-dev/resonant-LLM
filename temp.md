# ---------------- requirements.txt ----------------
requirements = """
# Core ML / NLP / Vision
torch>=2.2.0
torchvision>=0.17.0
torchaudio>=2.2.0
transformers>=4.43.0
sentence-transformers>=3.0.1
faiss-cpu>=1.8.0

# Audio IO + STT + TTS playback
sounddevice>=0.4.6
webrtcvad>=2.0.10
faster-whisper>=1.0.2
simpleaudio>=1.0.4

# Vision + GUI
opencv-python>=4.9.0.80
PyQt5>=5.15.10
Pillow>=10.2.0

# Utilities
numpy>=1.26.4
scipy>=1.13.0
tqdm>=4.66.4
pyyaml>=6.0.1
nltk>=3.8.1
"""

with open(os.path.join(root, "requirements.txt"), "w", encoding="utf-8") as f:
    f.write(textwrap.dedent(requirements).strip() + "\n")

# ---------------- config/config.yaml ----------------
config_yaml = """
# Fren-Agent Configuration
device: "cuda"   # "cuda" or "cpu"
model:
  # Vision captioning (BLIP)
  blip_model: "Salesforce/blip-image-captioning-base"
  # Local LLM backend for responses (GPT-2 sized) + Resonance gating
  text_model: "gpt2"
  max_new_tokens: 120
  temperature: 0.8
  top_p: 0.95

  # Resonance parameters (from Riemann resonance idea)
  resonance:
    sigma: 0.5       # Re(s), critical line locking
    tau: 14.134725   # Im(s) seed (can be updated dynamically)
    alpha: 1.5       # logits modulation strength
    primes: 97       # count of small primes to use

audio:
  sample_rate: 16000
  vad_frame_ms: 20
  vad_aggressiveness: 2
  stt_model_size: "base.en"  # faster-whisper size: tiny, base, small, medium, large-v3, etc.
  push_to_talk: true         # default PTT; set to false to enable VAD streaming

tts:
  engine: "piper"            # "piper" or "pyttsx3" as fallback
  piper:
    binary_path: "piper"     # system-resolvable 'piper' binary or absolute path
    model_path: ""           # e.g., C:/models/piper/en_GB-jenny_dioco.onnx
    output_wav: "./assets/tts_out.wav"

memory:
  embeddings_model: "sentence-transformers/all-MiniLM-L6-v2"
  index_path: "./assets/faiss_mem.index"
  metadata_path: "./assets/memory.json"
  k: 5

ui:
  camera_index: 0
  fps: 15
  theme: "dark"
"""

with open(os.path.join(config_dir, "config.yaml"), "w", encoding="utf-8") as f:
    f.write(textwrap.dedent(config_yaml).strip() + "\n")

# ---------------- src/fren_agent/__init__.py ----------------
init_py = """
__all__ = []
"""

with open(os.path.join(src, "__init__.py"), "w", encoding="utf-8") as f:
    f.write(init_py.strip() + "\n")

# ---------------- src/fren_agent/utils.py ----------------
utils_py = r'''
import os
import math
import json
import time
import hashlib
from typing import List

def ensure_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

def now_ms() -> int:
    return int(time.time() * 1000)

def sha_to_int(s: str) -> int:
    return int(hashlib.sha256(s.encode("utf-8")).hexdigest(), 16)

def small_primes(n: int) -> List[int]:
    """Return first n primes using simple sieve."""
    if n <= 0:
        return []
    # rough upper bound for nth prime ~ n (log n + log log n), use 20% cushion
    import math
    if n < 6:
        bound = 15
    else:
        bound = int(n * (math.log(n) + math.log(math.log(n))) * 1.2 + 10)
    sieve = [True] * (bound + 1)
    sieve[0] = sieve[1] = False
    for i in range(2, int(bound ** 0.5) + 1):
        if sieve[i]:
            step = i
            start = i * i
            sieve[start: bound + 1: step] = [False] * (((bound - start) // step) + 1)
    primes = [i for i, is_p in enumerate(sieve) if is_p]
    if len(primes) >= n:
        return primes[:n]
    # fallback grow
    x = bound + 1
    while len(primes) < n:
        is_p = True
        r = int(x ** 0.5) + 1
        for p in primes:
            if p > r:
                break
            if x % p == 0:
                is_p = False
                break
        if is_p:
            primes.append(x)
        x += 1
    return primes[:n]

def normalize(v):
    import numpy as np
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / (n + 1e-9)
'''

with open(os.path.join(src, "utils.py"), "w", encoding="utf-8") as f:
    f.write(utils_py.strip() + "\n")

# ---------------- src/fren_agent/memory.py ----------------
memory_py = r'''
import os
import json
import faiss
import numpy as np
from typing import List, Dict, Any

from sentence_transformers import SentenceTransformer

class SemanticMemory:
    def __init__(self, embeddings_model: str, index_path: str, metadata_path: str):
        self.model_name = embeddings_model
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.model = SentenceTransformer(self.model_name)
        self.dim = self.model.get_sentence_embedding_dimension()
        self.index = None
        self.meta: List[Dict[str, Any]] = []
        self._load()

    def _load(self):
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
        else:
            self.index = faiss.IndexFlatIP(self.dim)  # inner product with normalized vectors
            self.meta = []

    def _save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)

    def add(self, text: str, metadata: Dict[str, Any] = None):
        if metadata is None:
            metadata = {}
        emb = self.model.encode([text], normalize_embeddings=True)
        self.index.add(emb.astype(np.float32))
        self.meta.append({"text": text, "metadata": metadata})
        self._save()

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if self.index.ntotal == 0:
            return []
        emb = self.model.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, idxs = self.index.search(emb, k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0 or idx >= len(self.meta):
                continue
            item = dict(self.meta[idx])
            item["score"] = float(score)
            results.append(item)
        return results
'''

with open(os.path.join(src, "memory.py"), "w", encoding="utf-8") as f:
    f.write(memory_py.strip() + "\n")

# ---------------- src/fren_agent/emotion.py ----------------
emotion_py = r'''
from typing import Dict
import nltk

# Ensure VADER lexicon is available at runtime; if not, try to download.
def _ensure_vader():
    try:
        from nltk.sentiment import SentimentIntensityAnalyzer  # noqa: F401
        nltk.data.find('sentiment/vader_lexicon.zip')
    except Exception:
        nltk.download('vader_lexicon', quiet=True)

_ensure_vader()
from nltk.sentiment import SentimentIntensityAnalyzer

class EmotionAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def analyze_text(self, text: str) -> Dict[str, float]:
        scores = self.analyzer.polarity_scores(text or "")
        # Map to a simple mood indicator
        mood = "neutral"
        if scores["compound"] >= 0.35:
            mood = "positive"
        elif scores["compound"] <= -0.35:
            mood = "negative"
        scores["mood"] = mood
        return scores
'''

with open(os.path.join(src, "emotion.py"), "w", encoding="utf-8") as f:
    f.write(emotion_py.strip() + "\n")

# ---------------- src/fren_agent/vision.py ----------------
vision_py = r'''
from typing import Optional
import torch
from PIL import Image
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration

class VisionCaptioner:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device).eval()

    @torch.inference_mode()
    def caption_frame(self, frame_bgr: np.ndarray) -> str:
        # Convert BGR (OpenCV) to RGB PIL Image
        image = Image.fromarray(frame_bgr[:, :, ::-1])
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=40)
        txt = self.processor.decode(out[0], skip_special_tokens=True)
        return txt
'''

with open(os.path.join(src, "vision.py"), "w", encoding="utf-8") as f:
    f.write(vision_py.strip() + "\n")

# ---------------- src/fren_agent/audio.py ----------------
audio_py = r'''
import os
import wave
import queue
import threading
from dataclasses import dataclass
from typing import Optional

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
        self.device_info = sd.query_devices(kind="input")
        self.channels = 1

    def _callback(self, indata, frames, time, status):
        if status:
            # status may contain overruns/underruns
            pass
        # indata: float32; convert to int16
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
        num_frames = int((self.cfg.sample_rate * seconds) // (self.cfg.vad_frame_ms / 1000))
        raw = b""
        for _ in range(num_frames):
            try:
                raw += self.frames_q.get(timeout=1.0)
            except queue.Empty:
                break
        self.stop()
        self._write_wav(out_wav, raw, self.cfg.sample_rate)
        return out_wav

    def stream_and_segment(self, out_dir: str, silence_ms: int = 600, max_len_ms: int = 15000):
        """Yield wav paths for each VAD-detected utterance."""
        os.makedirs(out_dir, exist_ok=True)
        self.start()
        bytes_per_frame = int(self.cfg.sample_rate * (self.cfg.vad_frame_ms / 1000) * 2)  # 16-bit mono
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
                    # end of utterance
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
'''

with open(os.path.join(src, "audio.py"), "w", encoding="utf-8") as f:
    f.write(audio_py.strip() + "\n")

# ---------------- src/fren_agent/tts.py ----------------
tts_py = r'''
import os
import subprocess
import wave
import simpleaudio as sa
from typing import Optional

class PiperTTS:
    def __init__(self, binary_path: str, model_path: str, out_wav: str):
        self.binary_path = binary_path
        self.model_path = model_path
        self.out_wav = out_wav

    def speak(self, text: str):
        if not self.model_path:
            raise RuntimeError("Piper model_path not set.")
        # Piper expects text on stdin
        cmd = [self.binary_path, "-m", self.model_path, "-f", self.out_wav]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = proc.communicate(input=text.encode("utf-8"), timeout=60)
        if proc.returncode != 0:
            raise RuntimeError(f"Piper failed: {stderr.decode('utf-8', errors='ignore')}")
        self._play(self.out_wav)

    def _play(self, wav_path: str):
        with wave.open(wav_path, 'rb') as wf:
            data = wf.readframes(wf.getnframes())
            play_obj = sa.play_buffer(data, wf.getnchannels(), wf.getsampwidth(), wf.getframerate())
            play_obj.wait_done()

try:
    import pyttsx3
except Exception:
    pyttsx3 = None

class FallbackTTS:
    def __init__(self):
        if pyttsx3 is None:
            raise RuntimeError("pyttsx3 not available; install or configure Piper.")
        self.engine = pyttsx3.init()

    def speak(self, text: str):
        self.engine.say(text)
        self.engine.runAndWait()
'''

with open(os.path.join(src, "tts.py"), "w", encoding="utf-8") as f:
    f.write(tts_py.strip() + "\n")

# ---------------- src/fren_agent/llm/resonant_llm.py ----------------
resonant_llm_py = r'''
import math
from typing import Optional, List

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from ..utils import small_primes

class ResonantResponder:
    """
    Local text generation using a standard causal LM (e.g., GPT-2),
    with logits modulation inspired by Riemann resonance dynamics.

    The modulation computes a token-wise resonance score using a
    small set of primes and parameters (sigma, tau), then shifts
    logits before sampling. This implements an "entropy-symmetric"
    gating tendency around Re(s)=0.5.
    """
    def __init__(self, model_name: str = "gpt2", device: str = "cpu",
                 sigma: float = 0.5, tau: float = 14.134725, alpha: float = 1.5, primes: int = 97):
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device).eval()
        self.sigma = sigma
        self.tau = tau
        self.alpha = alpha
        self.primes = small_primes(primes)

    def _resonance_vector(self, vocab_size: int) -> torch.Tensor:
        """
        Compute a per-token resonance score vector r[i] of shape [vocab_size].
        Token index i acts as proxy for an integer 'n'. We form:
            r[i] = sum_{p in primes} w_p * phase(i, p)
        where w_p = p^{-sigma}, and phase flips sign based on congruence i mod p.
        We also include a tau-controlled oscillation via cos(tau * log(p)).
        """
        r = torch.zeros(vocab_size, dtype=torch.float32)
        for p in self.primes:
            w = (p ** (-self.sigma)) * math.cos(self.tau * math.log(p))
            # congruence pattern across vocab: tokens divisible by p get +1 else -1
            # Shifted so that roughly half tokens align, half anti-align.
            mask = torch.ones(vocab_size, dtype=torch.float32)
            # indices divisible by p -> +1; otherwise -1
            idx = torch.arange(vocab_size)
            mask[idx % p != 0] = -1.0
            r += w * mask
        # normalize
        r = torch.tanh(r / (len(self.pr