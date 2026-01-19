from __future__ import annotations
from dataclasses import dataclass
import re
from typing import Iterable

@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    source_path: str
    heading: str
    text: str

_heading_re = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)

def split_by_headings(md_text: str) -> list[tuple[str, str]]:
    """
    Returns list of (heading, section_text). If no headings, one section with heading "".
    """
    matches = list(_heading_re.finditer(md_text))
    if not matches:
        return [("", md_text)]

    sections: list[tuple[str, str]] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(md_text)
        heading = m.group(2).strip()
        section = md_text[start:end].strip()
        sections.append((heading, section))
    return sections

def tokenish_len(s: str) -> int:
    # cheap estimator; we’ll switch to tiktoken exact counts later
    return max(1, len(s) // 4)

def chunk_text(text: str, max_tokens: int = 420, overlap_tokens: int = 60) -> list[str]:
    """
    Sliding window chunking by token-ish length (fast MVP).
    """
    words = text.split()
    chunks = []
    start = 0
    # convert “tokens” to approximate words (1 token ~ 0.75 words-ish)
    max_w = int(max_tokens * 0.75)
    ov_w = int(overlap_tokens * 0.75)

    while start < len(words):
        end = min(len(words), start + max_w)
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        if end == len(words):
            break
        start = max(0, end - ov_w)
    return chunks

def make_chunks(docs: Iterable, max_tokens: int = 420, overlap_tokens: int = 60) -> list[Chunk]:
    out: list[Chunk] = []
    for d in docs:
        for heading, section in split_by_headings(d.text):
            for j, c in enumerate(chunk_text(section, max_tokens=max_tokens, overlap_tokens=overlap_tokens)):
                chunk_id = f"{d.doc_id}:{abs(hash((heading, j))) % (10**10)}"
                out.append(Chunk(
                    chunk_id=chunk_id,
                    doc_id=d.doc_id,
                    source_path=d.source_path,
                    heading=heading,
                    text=c
                ))
    return out
