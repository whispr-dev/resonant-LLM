import os
import json
import time
import uuid
import faiss
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

from sentence_transformers import SentenceTransformer
from .guardian import MemoryGuardian, GovernanceConfig

def _ts_ms() -> int:
    return int(time.time() * 1000)

class SemanticMemory:
    """
    Append-only (default) semantic memory with:
      - metadata fields: id, ts, source, protected, deleted, importance, novelty, tags
      - near-duplicate linking (no destructive dedupe)
      - soft-delete via tombstones (only if not protected and governance allows)
      - snapshot/restore for safety
    """
    def __init__(self, embeddings_model: str, index_path: str, metadata_path: str,
                 governance: Optional[Dict[str, Any]] = None):
        self.model_name = embeddings_model
        self.index_path = index_path
        self.metadata_path = metadata_path

        self.model = SentenceTransformer(self.model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

        self.index = None
        self.meta: List[Dict[str, Any]] = []
        self.guard = MemoryGuardian(GovernanceConfig.from_dict(governance or {}))
        self._load()

    # ---------- persistence ----------
    def _load(self):
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            self.index = faiss.read_index(self.index_path)
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)
        else:
            self.index = faiss.IndexFlatIP(self.dim)  # cosine if vectors are normalized
            self.meta = []

    def _save(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        with open(self.metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)

    # ---------- core ops ----------
    def _encode_norm(self, texts: List[str]) -> np.ndarray:
        emb = self.model.encode(texts, normalize_embeddings=True)
        return emb.astype(np.float32)

    def _nearest(self, emb: np.ndarray, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        if self.index.ntotal == 0:
            return np.zeros((1, k), dtype=np.float32), np.full((1, k), -1, dtype=np.int64)
        scores, idxs = self.index.search(emb, k)
        return scores, idxs

    def add(self, text: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Adds an item non-destructively.
        - Computes novelty vs top-k neighbors.
        - If not novel enough, links to primary; does NOT delete the new item.
        - Applies identity protection and importance heuristics.
        """
        metadata = dict(metadata or {})
        info = self.guard.classify(text)
        metadata.setdefault("tags", [])
        for t in info["tags"]:
            if t not in metadata["tags"]:
                metadata["tags"].append(t)
        protected = info["protected"]

        emb = self._encode_norm([text])
        # novelty: 1 - max cosine sim among top-5
        sim, idxs = self._nearest(emb, k=5)
        max_sim = float(sim[0].max()) if sim.size else 0.0
        novelty = 1.0 - max_sim

        # simple importance: longer texts + identity cue boosts
        length_bonus = min(len(text) / 200.0, 1.0)  # cap
        importance = length_bonus + (0.5 if protected else 0.0)
        importance = float(min(1.0, importance))

        store = self.guard.should_store(novelty=novelty, importance=importance)

        item = {
            "id": str(uuid.uuid4()),
            "text": text,
            "ts": _ts_ms(),
            "metadata": metadata,
            "protected": bool(protected),
            "deleted": False,
            "novelty": novelty,
            "importance": importance,
            "links": [],          # indices/ids of related near-dups
            "primary_of": []      # list of ids for which this is a primary (if any)
        }

        if store:
            # store vector + meta
            self.index.add(emb)
            self.meta.append(item)
            # link near-duplicates (non-destructive)
            self._link_near_dups(len(self.meta) - 1, idxs[0], sim[0])
            self._save()
        else:
            # still record non-stored entry in JSON for audit trail (no vector)
            item["not_stored_vector"] = True
            # link to closest neighbor id if exists
            if self.meta and int(idxs[0][0]) >= 0:
                closest_idx = int(idxs[0][0])
                neighbor = self.meta[closest_idx]
                item["links"].append({"id": neighbor["id"], "score": float(sim[0][0])})
                neighbor["primary_of"].append(item["id"])
            self.meta.append(item)
            self._save()

        return item

    def _link_near_dups(self, new_idx: int, neighbor_idxs: np.ndarray, neighbor_scores: np.ndarray):
        for idx, sc in zip(neighbor_idxs, neighbor_scores):
            if int(idx) < 0:
                continue
            if int(idx) == new_idx:
                continue
            if float(1.0 - float(sc)) < self.guard.cfg.min_novelty:
                n = self.meta[int(idx)]
                self.meta[new_idx]["links"].append({"id": n["id"], "score": float(sc)})
                # (optional) choose the older as primary
                if n["ts"] <= self.meta[new_idx]["ts"]:
                    n["primary_of"].append(self.meta[new_idx]["id"])
                else:
                    self.meta[new_idx]["primary_of"].append(n["id"])

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        if self.index.ntotal == 0:
            return []
        emb = self._encode_norm([query])
        scores, idxs = self.index.search(emb, k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0 or idx >= len(self.meta):
                continue
            m = self.meta[idx]
            if m.get("deleted"):
                continue
            item = dict(m)
            item["score"] = float(score)
            results.append(item)
        return results

    def soft_delete(self, entry_id: str, reason: str = "user_request") -> bool:
        """Tombstone an entry only if governance allows and not protected."""
        for m in self.meta:
            if m["id"] == entry_id:
                if self.guard.soft_delete_allowed(m):
                    m["deleted"] = True
                    m["tombstone_reason"] = reason
                    m["deleted_ts"] = _ts_ms()
                    self._save()
                    return True
                else:
                    return False
        return False

    def snapshot(self) -> str:
        return self.guard.snapshot(self.index_path, self.metadata_path)
