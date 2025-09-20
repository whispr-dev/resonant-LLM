from typing import Optional
import torch
from PIL import Image
import numpy as np
from transformers import BlipProcessor, BlipForConditionalGeneration

class VisionCaptioner:
    def __init__(self, model_name: str, device: str = "cpu"):
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(
            model_name,
            use_safetensors=True  # force safetensors; avoids torch.load on .bin
        ).to(self.device).eval()

    @torch.inference_mode()
    def caption_frame(self, frame_bgr: np.ndarray) -> str:
        image = Image.fromarray(frame_bgr[:, :, ::-1])  # BGR->RGB
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=40)
        txt = self.processor.decode(out[0], skip_special_tokens=True)
        return txt
