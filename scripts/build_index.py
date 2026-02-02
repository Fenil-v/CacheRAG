from __future__ import annotations
from pathlib import Path
import sys
import time
import orjson

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from rag.ingest import load_markdown_corpus
from rag.chunk import make_chunks
from rag.embed import Embedder
from rag.index_faiss import build_faiss_index, save_index, save_metadata

RAW_DIR = "data/raw"
OUT_META = "data/index/chunks.json"
OUT_INDEX = "data/index/faiss.index"

def main():
    t0 = time.time()
    docs = load_markdown_corpus(RAW_DIR)
    chunks = make_chunks(docs, max_tokens=420, overlap_tokens=60)

    texts = [c.text for c in chunks]
    meta = [{
        "chunk_id": c.chunk_id,
        "doc_id": c.doc_id,
        "source_path": c.source_path,
        "heading": c.heading,
        "text": c.text
    } for c in chunks]

    embedder = Embedder(device="cpu")  # Kaggle: set to "cuda" later
    t1 = time.time()
    embs = embedder.encode(texts, batch_size=64)
    t2 = time.time()

    index = build_faiss_index(embs)
    t3 = time.time()

    save_index(index, OUT_INDEX)
    save_metadata(meta, OUT_META)

    stats = {
        "docs": len(docs),
        "chunks": len(chunks),
        "embed_seconds": t2 - t1,
        "index_seconds": t3 - t2,
        "total_seconds": time.time() - t0
    }
    Path("experiments").mkdir(exist_ok=True)
    Path("experiments/build_stats.json").write_bytes(orjson.dumps(stats, option=orjson.OPT_INDENT_2))
    print(orjson.dumps(stats, option=orjson.OPT_INDENT_2).decode())

if __name__ == "__main__":
    main()
