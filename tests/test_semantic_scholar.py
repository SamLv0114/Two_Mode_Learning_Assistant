"""Semantic Scholar client: never raises, retries a rate limit, caches, and normalises ids."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx

from src.collectors import semantic_scholar as s2
from src.utils import redis_client

PAPER = {"paperId": "abc", "title": " Denoising Diffusion ", "abstract": "We present...", "year": 2020,
         "externalIds": {"ArXiv": "2006.11239v2"}, "authors": [{"name": "J. Ho"}],
         "publicationDate": "2020-06-19", "citationCount": 9000}


def _response(status, body=None):
    return SimpleNamespace(status_code=status, json=lambda: body)


class SemanticScholarTests(unittest.TestCase):
    def setUp(self):
        s2._cache.local.clear()
        patches = [patch.object(redis_client, "get_redis", return_value=None), patch.object(s2.time, "sleep"),
                   patch.object(s2, "MIN_INTERVAL_SECONDS", 0)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def test_arxiv_ids_lose_their_version_and_s2_ids_pass_through(self):
        self.assertEqual(s2.paper_ref("2006.11239v2"), "ArXiv:2006.11239")
        self.assertEqual(s2.paper_ref("649def34f8be52c8b66281af98ae884c09aef38b"),
                         "649def34f8be52c8b66281af98ae884c09aef38b")

    def test_parses_a_paper_and_its_arxiv_url(self):
        paper = s2.S2Paper.from_api(PAPER)
        self.assertEqual((paper.title, paper.arxiv_id, paper.authors), ("Denoising Diffusion", "2006.11239", ["J. Ho"]))
        self.assertEqual(paper.url, "https://arxiv.org/abs/2006.11239")
        self.assertIsNone(s2.S2Paper.from_api({"paperId": "x"}))  # no title

    def test_rate_limit_is_retried_then_cached(self):
        replies = [_response(429), _response(200, {"recommendedPapers": [PAPER]})]
        with patch.object(s2.httpx, "request", side_effect=lambda *a, **k: replies.pop(0)) as request:
            first = s2.recommend(["2006.11239"], ["2101.00001"])
            second = s2.recommend(["2006.11239"], ["2101.00001"])
        self.assertEqual([p.paper_id for p in first], ["abc"])
        self.assertEqual([p.paper_id for p in second], ["abc"])
        self.assertEqual(request.call_count, 2)  # the 429 and one success; the repeat came from the cache
        body = request.call_args.kwargs["json"]
        self.assertEqual(body, {"positivePaperIds": ["ArXiv:2006.11239"], "negativePaperIds": ["ArXiv:2101.00001"]})

    def test_no_answer_is_none_and_never_raises(self):
        with patch.object(s2.httpx, "request", side_effect=httpx.ConnectError("down")):
            self.assertIsNone(s2.search("diffusion"))
        with patch.object(s2.httpx, "request", return_value=_response(404)):
            self.assertIsNone(s2.references("2006.11239"))
        with patch.object(s2.httpx, "request", return_value=_response(429)) as request:
            self.assertIsNone(s2.recommend(["2006.11239"]))
        self.assertEqual(request.call_count, s2.RETRIES + 1)

    def test_an_answer_with_no_papers_is_an_empty_list(self):
        with patch.object(s2.httpx, "request", return_value=_response(200, {"data": []})):
            self.assertEqual(s2.search("nothing"), [])

    def test_published_falls_back_to_the_year(self):
        self.assertEqual(s2.S2Paper("x", "T", publication_date="2020-06-19").published.month, 6)
        self.assertEqual(s2.S2Paper("x", "T", year=2019).published.year, 2019)
        self.assertIsNone(s2.S2Paper("x", "T").published)

    def test_no_positives_means_no_call(self):
        with patch.object(s2.httpx, "request") as request:
            self.assertEqual(s2.recommend([]), [])
        request.assert_not_called()

    def test_references_read_the_cited_paper(self):
        with patch.object(s2.httpx, "request", return_value=_response(200, {"data": [{"citedPaper": PAPER}]})):
            self.assertEqual([p.title for p in s2.references("2301.00001")], ["Denoising Diffusion"])


if __name__ == "__main__":
    unittest.main()
