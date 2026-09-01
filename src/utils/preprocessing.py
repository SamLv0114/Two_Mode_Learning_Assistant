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


def extract_text_from_pdf(data: bytes) -> str:
    """Extract text from a PDF byte stream using PyMuPDF."""
    if _fitz is None:
        raise RuntimeError("PyMuPDF is not installed; PDF extraction is unavailable.")

    doc = _fitz.open(stream=data, filetype="pdf")
    return "\n".join(page.get_text() for page in doc).strip()



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

