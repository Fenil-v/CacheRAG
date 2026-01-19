from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib

@dataclass(frozen=True)
class Doc:
    doc_id: str
    source_path: str
    text: str

def _stable_id(path: Path) -> str:
    h = hashlib.sha1(str(path).encode("utf-8")).hexdigest()
    return h[:16]

def load_markdown_corpus(raw_dir: str) -> list[Doc]:
    base = Path(raw_dir)
    paths = sorted([p for p in base.rglob("*.md") if p.is_file()])
    docs: list[Doc] = []
    for p in paths:
        txt = p.read_text(encoding="utf-8", errors="ignore")
        docs.append(Doc(
            doc_id=_stable_id(p),
            source_path=str(p),
            text=txt
        ))
    return docs
