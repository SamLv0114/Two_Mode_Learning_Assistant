"""
Data preprocessing utilities
"""
from typing import List, Dict
import io
import re
from bs4 import BeautifulSoup

try:
    import fitz as _fitz  # PyMuPDF
except Exception:  # Optional dependency for PDF support
    _fitz = None


def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:()\[\]{}\-]', '', text)
    
    return text.strip()


def extract_text_from_html(html: str) -> str:
    """Extract clean text from HTML"""
    soup = BeautifulSoup(html, "html.parser")
    
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    
    # Get text
    text = soup.get_text()
    
    # Clean up whitespace
    lines = (line.strip() for line in text.splitlines())
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    text = " ".join(chunk for chunk in chunks if chunk)
    
    return text


def extract_text_from_pdf(data: bytes, table_aware: bool | None = None) -> str:
    """Extract paragraph text and detected tables from a PDF byte stream."""
    if _fitz is None:
        raise RuntimeError("PyMuPDF is not installed; PDF extraction is unavailable.")

    if table_aware is None:
        from src.utils.config import settings
        table_aware = settings.RAG_TABLE_EXTRACTION_ENABLED

    doc = _fitz.open(stream=data, filetype="pdf")
    if not table_aware:
        try:
            return "\n".join(page.get_text() for page in doc).strip()
        finally:
            doc.close()
    pages = []
    for page in doc:
        tables = []
        try:
            tables = list(page.find_tables().tables)
        except (AttributeError, ValueError):
            pass
        table_rects = [_fitz.Rect(table.bbox) for table in tables]
        paragraphs = []
        for block in page.get_text("blocks"):
            rect = _fitz.Rect(block[:4])
            if any((rect & table_rect).get_area() >= 0.5 * max(rect.get_area(), 1) for table_rect in table_rects):
                continue
            text = str(block[4]).strip()
            if text:
                paragraphs.append(text)
        for table in tables:
            rows = table.extract() or []
            if rows:
                cells = [[str(cell or "").replace("|", "\\|").replace("\n", " ") for cell in row] for row in rows]
                paragraphs.append("\n".join("| " + " | ".join(row) + " |" for row in cells))
        pages.append("\n\n".join(paragraphs))
    doc.close()
    return "\n\n".join(pages).strip()


def normalize_document_text(text: str) -> str:
    """Normalize whitespace without flattening paragraphs or Markdown tables."""
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in (text or "").splitlines()]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def semantic_parent_child_chunks(text: str, parent_size: int = 1800,
                                 child_size: int = 500, overlap: int = 50) -> List[Dict]:
    """Group paragraphs into parents; split children at sentence/line edges."""
    if parent_size < child_size or child_size <= 0 or not 0 <= overlap < child_size:
        raise ValueError("Require parent_size >= child_size > overlap >= 0")
    text = normalize_document_text(text)
    if not text:
        return []
    units = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
    parents = []
    current = ""
    for unit in units:
        if current and len(current) + len(unit) + 2 > parent_size:
            parents.append(current)
            current = ""
        if len(unit) > parent_size:
            if current:
                parents.append(current)
                current = ""
            while len(unit) > parent_size:
                cut = max(unit.rfind(". ", 0, parent_size), unit.rfind("\n", 0, parent_size))
                cut = cut + 1 if cut > parent_size // 2 else parent_size
                parents.append(unit[:cut].strip())
                unit = unit[cut:].strip()
        current = f"{current}\n\n{unit}".strip() if current else unit
    if current:
        parents.append(current)

    chunks = []
    for parent_index, parent in enumerate(parents):
        start = 0
        while start < len(parent):
            end = min(start + child_size, len(parent))
            if end < len(parent):
                edge = max(parent.rfind(". ", start, end), parent.rfind("\n", start, end))
                if edge > start + child_size // 2:
                    end = edge + 1
            child = parent[start:end].strip()
            if child:
                chunks.append({"child": child, "parent": parent, "parent_index": parent_index})
            if end == len(parent):
                break
            start = max(start + 1, end - overlap)
    return chunks



def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    
    return chunks

