from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import faiss
import numpy as np
import orjson

def build_faiss_index(embeddings: np.ndarray) -> faiss.Index:
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine sim because normalized
    index.add(embeddings)
    return index

def save_index(index: faiss.Index, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, path)

def load_index(path: str) -> faiss.Index:
    return faiss.read_index(path)

def save_metadata(meta: list[dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(orjson.dumps(meta))

def load_metadata(path: str) -> list[dict]:
    return orjson.loads(Path(path).read_bytes())
