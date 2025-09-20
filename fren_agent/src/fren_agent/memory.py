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
            self.index = faiss.IndexFlatIP(self.dim)
            self.meta = []

    def _save(self):
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
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
