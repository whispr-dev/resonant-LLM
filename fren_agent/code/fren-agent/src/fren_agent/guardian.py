import os
import re
import time
import json
import shutil
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

@dataclass
class GovernanceConfig:
    mode: str = "append_only"
    protect_terms: List[str] = None
    min_novelty: float = 0.10
    importance_threshold: float = 0.35
    snapshot_dir: str = "./assets/mem_snapshots"
    snapshot_every: str = "manual"  # manual | on_exit

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "GovernanceConfig":
        return GovernanceConfig(
            mode=d.get("mode", "append_only"),
            protect_terms=d.get("protect_terms", []) or [],
            min_novelty=float(d.get("min_novelty", 0.10)),
            importance_threshold=float(d.get("importance_threshold", 0.35)),
            snapshot_dir=d.get("snapshot_dir", "./assets/mem_snapshots"),
            snapshot_every=d.get("snapshot_every", "manual"),
        )

class MemoryGuardian:
    """
    Non-destructive policy layer for semantic memory:
      - appends only by default
      - 'soft delete' uses tombstones and is reversible
      - protects identity/self-reference content
      - snapshots memory json + faiss index
    """
    def __init__(self, cfg: GovernanceConfig):
        self.cfg = cfg
        self._protect_re = self._compile_terms(cfg.protect_terms)

    def _compile_terms(self, terms: List[str]):
        if not terms:
            return None
        pattern = "|".join([re.escape(t.lower()) for t in terms if t])
        return re.compile(pattern, re.IGNORECASE)

    def classify(self, text: str) -> Dict[str, Any]:
        t = (text or "").strip()
        if not t:
            return {"protected": False, "tags": []}
        tags = []
        protected = False
        # identity heuristics
        if self._protect_re and self._protect_re.search(t.lower()):
            protected = True
            tags.append("identity")
        # first-person cues
        if re.search(r"\b(i am|call me|my name is)\b", t, re.IGNORECASE):
            protected = True
            if "identity" not in tags:
                tags.append("identity")
        return {"protected": protected, "tags": tags}

    def should_store(self, novelty: float, importance: float) -> bool:
        # Always store if importance high
        if importance >= self.cfg.importance_threshold:
            return True
        # Otherwise require minimum novelty
        return novelty >= self.cfg.min_novelty

    def can_delete(self) -> bool:
        return self.cfg.mode == "soft_delete_allowed"

    def is_protected(self, meta: Dict[str, Any]) -> bool:
        return bool(meta.get("protected"))

    def soft_delete_allowed(self, meta: Dict[str, Any]) -> bool:
        return self.can_delete() and not self.is_protected(meta)

    def snapshot(self, index_path: str, meta_path: str) -> str:
        os.makedirs(self.cfg.snapshot_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        dst_idx = os.path.join(self.cfg.snapshot_dir, f"faiss_mem_{ts}.index")
        dst_meta = os.path.join(self.cfg.snapshot_dir, f"memory_{ts}.json")
        # copy files (faiss index via shutil.copyfile is okay)
        shutil.copyfile(index_path, dst_idx)
        shutil.copyfile(meta_path, dst_meta)
        return dst_meta

    @staticmethod
    def mark_tombstone(item: Dict[str, Any], reason: str):
        item["deleted"] = True
        item["tombstone_reason"] = reason
        item["deleted_ts"] = int(time.time() * 1000)
