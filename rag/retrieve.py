from __future__ import annotations
import numpy as np
import faiss

def search(index: faiss.Index, q_emb: np.ndarray, top_k: int = 5):
    if q_emb.ndim == 1:
        q_emb = q_emb[None, :]
    scores, ids = index.search(q_emb.astype("float32"), top_k)
    return scores[0].tolist(), ids[0].tolist()
