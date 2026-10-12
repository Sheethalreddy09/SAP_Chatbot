"""Bounded PDF page context for existing indexes; does not change stored vectors."""
import hashlib
import logging
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

import pymupdf

logger = logging.getLogger(__name__)
MAX_PAGE_CHARS = 10000
MAX_CONTEXT_CHARS = 32000


def _normalize(text):
    text = re.sub(r"[\ue000-\uf8ff\ufffd]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


@lru_cache(maxsize=32)
def _file_hash(path, mtime_ns, size):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=128)
def _read_page(path, mtime_ns, size, page_number):
    with pymupdf.open(path) as document:
        if not 1 <= page_number <= len(document):
            raise ValueError("Page outside document")
        page = document[page_number - 1]
        text = page.get_text("text") or ""
        blocks = page.get_text("dict")["blocks"]
    sizes = Counter()
    for block in blocks:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                sizes[round(span["size"], 1)] += len(span["text"].strip())
    body_size = sizes.most_common(1)[0][0] if sizes else 9.0
    headings = []
    # Typography supplies evidence of headings; uppercase alone does not.
    for block in blocks:
        heading_lines = []
        for line in block.get("lines", []):
            spans = [s for s in line.get("spans", []) if s["text"].strip()]
            if spans and min(s["size"] for s in spans) >= body_size * 1.4:
                value = _normalize(" ".join(s["text"] for s in spans))
                if value.upper() not in {"PUBLIC", "INTERNAL", "CONFIDENTIAL"}:
                    heading_lines.append(value)
            elif heading_lines:
                headings.append(" ".join(heading_lines))
                heading_lines = []
        if heading_lines:
            headings.append(" ".join(heading_lines))
    return text.strip(), tuple(dict.fromkeys(headings))


def expand_page_context(chunks, roots, expected_hashes):
    """Expand only unchanged indexed PDFs whose cited passage matches the page.

    Keep retrieval scores untouched. Group same-page hits only after successfully
    expanding the entire page. Missing files or oversized pages retain the original
    passage with an explicit incomplete-context label. No passage is dropped.
    """
    result, expanded, used_chars = [], set(), 0
    for chunk in chunks:
        item = dict(chunk)
        original = item.get("chunk_text", "")
        item["scope_context_status"] = "passage_only"
        item["section"] = "Section unverified; retrieved passage only"
        name = item.get("document_name", "")
        page = item.get("page_number")
        key = (item.get("document_id"), name, page)
        if key in expanded:
            continue
        try:
            # Reject path components, then resolve inside the permitted KB roots.
            if not name or "/" in name or "\\" in name or Path(name).suffix.lower() != ".pdf":
                raise ValueError("Not a PDF basename")
            expected = expected_hashes.get(item.get("document_id"))
            if not expected:
                raise ValueError("Indexed hash unavailable")
            candidates = set()
            for root in roots:
                root = Path(root).resolve()
                if not root.is_dir():
                    continue
                for candidate in root.rglob(name):
                    resolved = candidate.resolve()
                    if resolved.is_relative_to(root) and resolved.is_file():
                        candidates.add(resolved)
            if len(candidates) != 1:
                raise ValueError("Missing or ambiguous source file")
            path = candidates.pop()
            stat = path.stat()
            cache_key = (str(path), stat.st_mtime_ns, stat.st_size)
            if _file_hash(*cache_key) != expected:
                raise ValueError("Source changed since indexing")
            text, headings = _read_page(*cache_key, int(page))
            # A full normalized match avoids mixing an unrelated passage/page.
            if not original.strip() or _normalize(original) not in _normalize(text):
                raise ValueError("Indexed passage does not match source page")
            if len(text) > MAX_PAGE_CHARS or used_chars + len(text) > MAX_CONTEXT_CHARS:
                raise ValueError("Expansion budget reached")
            item["chunk_text"] = text
            item["section"] = (
                "Page headings: " + " / ".join(headings)
                if headings else "Page context; section heading not established"
            )
            item["scope_context_status"] = "full_page; may contain multiple sections"
            item["retrieved_passage"] = original
            used_chars += len(text)
            expanded.add(key)
        except (OSError, ValueError, RuntimeError, TypeError) as exc:
            logger.info("Keeping indexed passage without page expansion: %s", exc)
        result.append(item)
    return result


def prepare_scoped_evidence(db, chunks):
    # Import lazily so pure page-context tests do not start the app/database.
    from backend.app.config import settings
    from backend.app.models.document import Document

    ids = {c.get("document_id") for c in chunks if c.get("document_id")}
    documents = db.query(Document).filter(Document.id.in_(ids)).all() if ids else []
    return expand_page_context(
        chunks,
        [settings.KNOWLEDGE_BASE_DIR, settings.DOCUMENTS_DIR],
        {d.id: d.file_hash for d in documents},
    )