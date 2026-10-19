"""Long abstracts are embedded in pieces and averaged, so text past the model's input limit counts."""
import unittest

import numpy as np

from src.models.embeddings import EmbeddingManager, _scope_where


class _Tokenizer:
    """One token per word; ids are word positions in a shared vocabulary."""

    def __init__(self):
        self.vocab = {}

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [self.vocab.setdefault(word, len(self.vocab)) for word in text.split()]}

    def decode(self, ids):
        words = {i: w for w, i in self.vocab.items()}
        return " ".join(words[i] for i in ids)


class _Model:
    def __init__(self, limit):
        self.tokenizer = _Tokenizer()
        self.max_seq_length = limit + 2
        self.encoded = []

    def encode(self, texts, **kwargs):
        self.encoded.extend(texts)
        # A unit vector per piece: [1, 0] when it mentions "late", [0, 1] otherwise.
        return np.array([[1.0, 0.0] if "late" in t else [0.0, 1.0] for t in texts])


def manager(limit):
    em = EmbeddingManager.__new__(EmbeddingManager)
    em.model = _Model(limit)
    return em


class PaperPieceTests(unittest.TestCase):
    def test_short_paper_is_one_piece(self):
        self.assertEqual(manager(50).paper_pieces("Title", "a short abstract"), ["Title\n\na short abstract"])

    def test_long_abstract_is_split_and_every_piece_starts_with_the_title(self):
        em = manager(20)
        abstract = " ".join(f"w{i}" for i in range(60))
        pieces = em.paper_pieces("Title", abstract)
        self.assertGreater(len(pieces), 2)
        self.assertTrue(all(p.startswith("Title") for p in pieces))
        self.assertTrue(all(len(p.split()) <= 20 for p in pieces))
        self.assertIn("w59", pieces[-1])  # nothing past the limit is dropped

    def test_pooled_vector_counts_the_part_a_truncated_one_never_saw(self):
        em = manager(10)
        abstract = " ".join(["early"] * 15 + ["late"] * 15)
        vector = em.embed_papers(["T"], [abstract])[0]
        self.assertGreater(vector[0], 0.1)  # the "late" pieces pull the mean their way
        self.assertAlmostEqual(float(np.linalg.norm(vector)), 1.0, places=6)

    def test_one_vector_per_paper_in_order(self):
        em = manager(50)
        vectors = em.embed_papers(["A", "B"], ["late result", "early result"])
        self.assertEqual(len(vectors), 2)
        self.assertEqual([round(v[0]) for v in vectors], [1, 0])


class SearchScopeTests(unittest.TestCase):
    """What an unfiltered search may return: whole papers and the requester's own uploads, nothing else."""

    def test_unfiltered_search_covers_papers_and_own_uploads_only(self):
        self.assertEqual(_scope_where(None, 7), {"$or": [{"type": "paper"},
                                                         {"$and": [{"type": "user_doc"}, {"user_id": 7}]}]})

    def test_without_a_user_only_papers(self):
        self.assertEqual(_scope_where(None, None), {"type": "paper"})

    def test_uploads_without_a_user_match_nothing(self):
        self.assertEqual(_scope_where("user_doc", None), {"$and": [{"type": "user_doc"}, {"user_id": -1}]})

    def test_explicit_type(self):
        self.assertEqual(_scope_where("paper", 7), {"type": "paper"})


if __name__ == "__main__":
    unittest.main()
