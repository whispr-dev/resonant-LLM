import math
from typing import Optional, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from ..utils import small_primes

class ResonantResponder:
    """
    Local text generation using a standard causal LM (e.g., GPT-2),
    with logits modulation inspired by Riemann resonance dynamics.

    The modulation computes a token-wise resonance score using a
    set of small primes and parameters (sigma, tau), then shifts
    logits before sampling. This implements an entropy-symmetric
    gating tendency around Re(s) ~ 1/2.
    """
    def __init__(self, model_name: str = "gpt2", device: str = "cpu",
                 sigma: float = 0.5, tau: float = 14.134725, alpha: float = 1.5, primes: int = 97):
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.tok = AutoTokenizer.from_pretrained(model_name)
        # ensure pad token if missing (gpt2 quirk)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device).eval()
        self.sigma = float(sigma)
        self.tau = float(tau)
        self.alpha = float(alpha)
        self.primes = small_primes(int(primes))
        self._cached_vec: Optional[torch.Tensor] = None
        self._cached_vocab: Optional[int] = None

    def _resonance_vector(self, vocab_size: int) -> torch.Tensor:
        """
        Compute a per-token resonance score vector r[i] of shape [vocab_size].
        Token index i ~ integer n. We form:
            r[i] = sum_{p in primes} w_p * phase(i, p)
        where w_p = p^{-sigma} * cos(tau * log p), and phase flips sign based on i mod p.
        """
        if self._cached_vec is not None and self._cached_vocab == vocab_size:
            return self._cached_vec

        idx = torch.arange(vocab_size, dtype=torch.int64)
        r = torch.zeros(vocab_size, dtype=torch.float32)
        for p in self.primes:
            w = (p ** (-self.sigma)) * math.cos(self.tau * math.log(p))
            mask = torch.ones_like(idx, dtype=torch.float32)
            mask[(idx % p) != 0] = -1.0
            r += w * mask
        r = torch.tanh(r / (len(self.primes) ** 0.5))
        self._cached_vec = r.to(self.device)
        self._cached_vocab = vocab_size
        return self._cached_vec

    @staticmethod
    def _entropy(logits: torch.Tensor) -> float:
        probs = torch.softmax(logits, dim=-1)
        ent = -(probs * torch.log(probs + 1e-9)).sum(dim=-1)
        return float(ent.detach().cpu().item())

    def _modulate(self, logits: torch.Tensor) -> torch.Tensor:
        """Add resonance vector scaled by alpha. Shape: [vocab]."""
        vec = self._resonance_vector(logits.size(-1))
        return logits + self.alpha * vec

    @torch.inference_mode()
    def generate(self,
                 prompt: str,
                 max_new_tokens: int = 120,
                 temperature: float = 0.8,
                 top_p: float = 0.95,
                 memory_snippets: Optional[List[str]] = None) -> Tuple[str, dict]:
        """
        Generate with resonance-modulated logits and optional memory snippets as context.
        Returns (text, diagnostics).
        """
        if memory_snippets:
            memory_block = "\n\n[MEMORY]\n" + "\n".join(f"- {m}" for m in memory_snippets) + "\n\n"
        else:
            memory_block = ""

        full_prompt = memory_block + prompt
        toks = self.tok(full_prompt, return_tensors="pt").to(self.device)
        input_ids = toks["input_ids"]

        out_tokens: List[int] = []
        diag = {"steps": []}

        for _ in range(int(max_new_tokens)):
            outputs = self.model(input_ids=input_ids)
            logits = outputs.logits[:, -1, :].squeeze(0)  # [vocab]
            base_entropy = self._entropy(logits)

            # resonance modulation
            logits = self._modulate(logits)

            # temperature + nucleus
            if temperature > 0:
                logits = logits / max(1e-6, temperature)
            probs = torch.softmax(logits, dim=-1)
            if 0.0 < top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(sorted_probs, dim=-1)
                mask = cumsum <= top_p
                # keep at least 1
                if not mask.any():
                    mask[0] = True
                filtered_idx = sorted_idx[mask]
                filtered_probs = sorted_probs[mask]
                filtered_probs = filtered_probs / filtered_probs.sum()
                next_id = filtered_idx[torch.multinomial(filtered_probs, num_samples=1)]
            else:
                next_id = torch.multinomial(probs, num_samples=1)
            next_id_int = int(next_id.item())
            out_tokens.append(next_id_int)

            # diagnostics (optional)
            diag["steps"].append({
                "base_entropy": base_entropy,
                "chosen_id": next_id_int
            })

            input_ids = torch.cat([input_ids, next_id.view(1, 1)], dim=1)

            if next_id_int == self.tok.eos_token_id:
                break

            # mild “dynamic tau” wobble using current position
            self.tau += 0.005 * math.cos(0.1 * len(out_tokens))

        text = self.tok.decode(out_tokens, skip_special_tokens=True)
        return text, diag
