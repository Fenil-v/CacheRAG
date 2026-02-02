from __future__ import annotations
import sys
from pathlib import Path
import time

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rag.embed import Embedder
from rag.index_faiss import load_index, load_metadata
from rag.retrieve import search

INDEX_PATH = "data/index/faiss.index"
META_PATH = "data/index/chunks.json"

def main():
    index = load_index(INDEX_PATH)
    meta = load_metadata(META_PATH)
    embedder = Embedder(device="cpu")

    while True:
        q = input("\nQuery (enter to quit): ").strip()
        if not q:
            break

        t0 = time.time()
        q_emb = embedder.encode([q], batch_size=1)[0]
        t1 = time.time()
        scores, ids = search(index, q_emb, top_k=5)
        t2 = time.time()

        print(f"\nembed_ms={(t1-t0)*1000:.2f}  retrieve_ms={(t2-t1)*1000:.2f}")
        for rank, (s, idx) in enumerate(zip(scores, ids), start=1):
            if idx < 0: 
                continue
            m = meta[idx]
            print(f"\n[{rank}] score={s:.4f}  {m['source_path']}  heading={m['heading']}")
            print(m["text"][:600] + ("..." if len(m["text"]) > 600 else ""))

if __name__ == "__main__":
    main()
