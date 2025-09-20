import subprocess
import wave
import simpleaudio as sa

try:
    import pyttsx3
except Exception:
    pyttsx3 = None

class PiperTTS:
    def __init__(self, binary_path: str, model_path: str, out_wav: str):
        self.binary_path = binary_path
        self.model_path = model_path
        self.out_wav = out_wav

    def speak(self, text: str):
        if not self.model_path:
            raise RuntimeError("Piper model_path not set.")
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

class FallbackTTS:
    def __init__(self):
        if pyttsx3 is None:
            raise RuntimeError("pyttsx3 not available; install or configure Piper.")
        self.engine = pyttsx3.init()

    def speak(self, text: str):
        self.engine.say(text)
        self.engine.runAndWait()
