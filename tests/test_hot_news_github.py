"""GitHub side of What's Hot: Trending-page parsing, AI filtering, fallback."""
import unittest
from unittest.mock import patch

from src.agents.hot_news_collector import HotNewsCollector, parse_trending_html


def _row(full_name, description, total, gained, language="Python"):
    return f'''
<article class="Box-row">
  <h2 class="h3 lh-condensed">
    <a data-hydro-click="{{}}" href="/{full_name}" class="Link"><svg></svg>
      <span class="text-normal">{full_name.split("/")[0]} /</span> {full_name.split("/")[1]}</a>
  </h2>
  <p class="col-9 color-fg-muted my-1 tmp-pr-4">
    {description}
  </p>
  <div class="f6 color-fg-muted mt-2">
    <span><span itemprop="programmingLanguage">{language}</span></span>
    <a href="/{full_name}/stargazers" class="Link"><svg></svg>
      {total}</a>
    <a href="/{full_name}/forks" class="Link"><svg></svg>
      12</a>
    <span class="d-inline-block float-sm-right"><svg></svg>
      {gained} stars this week
    </span>
  </div>
</article>'''


PAGE = "<html>" + "".join([
    _row("acme/agent-kit", "A toolkit for building LLM agents &amp; tools", "12,345", "2,001"),
    _row("acme/gym-log", "Self-hosted workout tracker", "9,000", "5,000", "JavaScript"),
    _row("big/rag-server", "Production RAG server", "800", "4,100"),
]) + "</html>"


def _new_repo(name):
    return {"name": name, "full_name": f"new/{name}", "description": "", "stars": 50,
            "stars_gained": None, "language": "", "url": "", "source": "GitHub", "trend_basis": "new"}


class TrendingParseTests(unittest.TestCase):
    def test_reads_every_field(self):
        repo = parse_trending_html(PAGE)[0]
        self.assertEqual(repo["full_name"], "acme/agent-kit")
        self.assertEqual(repo["name"], "agent-kit")
        self.assertEqual(repo["description"], "A toolkit for building LLM agents & tools")
        self.assertEqual((repo["stars"], repo["stars_gained"]), (12345, 2001))
        self.assertEqual(repo["language"], "Python")
        self.assertEqual(repo["url"], "https://github.com/acme/agent-kit")
        self.assertEqual(repo["trend_basis"], "trending")

    def test_unknown_layout_yields_nothing(self):
        self.assertEqual(parse_trending_html("<html><div>redesigned</div></html>"), [])


class TrendingSelectionTests(unittest.TestCase):
    def test_ranks_ai_repos_by_stars_gained(self):
        collector = HotNewsCollector()
        with patch.object(collector, "_scrape_trending", return_value=parse_trending_html(PAGE)), \
                patch.object(collector, "_search_new_repos", return_value=[]) as search:
            repos = collector._fetch_github_trending("weekly", limit=2)
        # The gym tracker gained the most stars but is not AI-related.
        self.assertEqual([r["full_name"] for r in repos], ["big/rag-server", "acme/agent-kit"])
        search.assert_not_called()

    def test_tops_up_from_new_repos_when_trending_is_short(self):
        collector = HotNewsCollector()
        with patch.object(collector, "_scrape_trending", return_value=parse_trending_html(PAGE)), \
                patch.object(collector, "_search_new_repos", return_value=[_new_repo("a"), _new_repo("b")]):
            repos = collector._fetch_github_trending("weekly", limit=3)
        self.assertEqual([r["trend_basis"] for r in repos], ["trending", "trending", "new"])

    def test_falls_back_entirely_when_trending_unreachable(self):
        collector = HotNewsCollector()
        with patch.object(collector, "_scrape_trending", return_value=[]), \
                patch.object(collector, "_search_new_repos", return_value=[_new_repo("a")]):
            repos = collector._fetch_github_trending("daily", limit=5)
        self.assertEqual([r["full_name"] for r in repos], ["new/a"])


if __name__ == "__main__":
    unittest.main()
