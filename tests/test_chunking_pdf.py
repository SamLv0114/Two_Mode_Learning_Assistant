"""Table extraction and parent-child evidence retrieval fixtures."""

import unittest
from unittest.mock import patch

import fitz

from src.utils.preprocessing import extract_text_from_pdf, semantic_parent_child_chunks
from src.models.embeddings import EmbeddingManager


class ChunkingPdfTests(unittest.TestCase):
    def test_semantic_children_keep_parent_paragraph(self):
        text = "First paragraph explains RRF fusion. It combines rankings.\n\nSecond paragraph details reranking. It scores candidates."
        records = semantic_parent_child_chunks(text, parent_size=70, child_size=38, overlap=5)
        self.assertTrue(records)
        self.assertTrue(all(record["child"] in record["parent"] for record in records))
        self.assertGreaterEqual(len({record["parent_index"] for record in records}), 2)
        self.assertTrue(all(len(record["child"]) <= 38 for record in records))

    def test_pdf_table_is_retained_as_rows(self):
        pdf = fitz.open()
        page = pdf.new_page()
        page.insert_text((40, 40), "Methods section.")
        for x in (40, 140, 240):
            page.draw_line((x, 80), (x, 160))
        for y in (80, 120, 160):
            page.draw_line((40, y), (240, y))
        for x, y, value in ((50, 100, "Method"), (150, 100, "Score"),
                            (50, 140, "RRF"), (150, 140, "0.82")):
            page.insert_text((x, y), value)
        extracted = extract_text_from_pdf(pdf.tobytes(), table_aware=True)
        baseline = extract_text_from_pdf(pdf.tobytes(), table_aware=False)
        pdf.close()
        self.assertIn("Methods section.", extracted)
        self.assertIn("| Method | Score |", extracted)
        self.assertIn("| RRF | 0.82 |", extracted)
        self.assertNotIn("| RRF | 0.82 |", baseline)

    def test_fixed_chunking_remains_default_until_full_corpus_evidence(self):
        text = "First paragraph. " * 70 + "\n\n" + "Second paragraph. " * 70
        with patch("src.models.embeddings.settings.RAG_SEMANTIC_CHUNKING_ENABLED", False):
            fixed = EmbeddingManager._chunk_records(text)
        with patch("src.models.embeddings.settings.RAG_SEMANTIC_CHUNKING_ENABLED", True):
            semantic = EmbeddingManager._chunk_records(text)
        self.assertTrue(all(row["parent"] == row["child"] for row in fixed))
        self.assertTrue(any(row["parent"] != row["child"] for row in semantic))


if __name__ == "__main__":
    unittest.main()
