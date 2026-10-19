"""
Tuning constants and pure helpers shared by the feed modules.

Ranking weights, penalty strengths, category calibration and the focus-area
vocabulary live here so retrieval and ranking read the same values.
"""
from typing import Dict, List

from src.utils.config import settings

# Ranking weights are derived from one continuous dial rather than three discrete
# modes. The old learning/frontier/balanced buckets re-derived the same
# authority-vs-recency axis the user already picks explicitly at the Recommended /
# Latest toggle, and a hard threshold on it (recent_ratio > 0.7) could flip a feed
# between very different results after a handful of interactions.
#
# Semantic similarity is how well a paper matches *this* user, so it stays fixed —
# it is the point of personalisation. What actually trades off is how the remaining
# budget is split between established influence and freshness.
SEMANTIC_WEIGHT = 0.55

TRADEOFF_BUDGET = 0.45  # split between citation and recency by the novelty dial

# Multiplicative demotions applied after the weighted score. Both cap well below
# 1.0 so a penalised paper can still surface if nothing better exists — these are
# preferences, not filters.
# Interactions needed before the trained ranker replaces heuristic scoring.
# Named because it gates four separate branches; a literal in each was how one of
# them (the impact-score fallback) ended up never checking it at all.
LIGHTGBM_MIN_INTERACTIONS = settings.MIN_INTERACTIONS_FOR_TRAINING

FAMILIAR_PENALTY_STRENGTH = 0.30   # similar to something already saved

# A dismiss only says the user rejected one paper; it does not reveal why.
# Similarity to that paper's title and abstract is therefore a weak topic hint,
# rather than a strong negative preference for every paper on that topic.
DISLIKE_PENALTY_STRENGTH = 0.08  # one dismiss does not establish a firm preference

# Calibrated selection (Steck, "Calibrated Recommendations", RecSys 2018): the
# final top-K should reflect the user's own category mix, not just whichever
# category the profile-driven semantic score currently favors. Declared
# interests act as a prior — expressed as CALIBRATION_PRIOR_STRENGTH "virtual"
# saved papers split across whichever declared areas map to a known category —
# so a category can't be silently zeroed out just because the user hasn't
# happened to save anything in it yet. Saved/viewed papers' real arXiv
# categories are the evidence that shifts the target as it accumulates.
CALIBRATION_PRIOR_STRENGTH = 5.0    # prior weight, in "papers", split across declared interests

CALIBRATION_EVIDENCE_MAX_IDS = 200  # recent saved/viewed papers considered as evidence

# Candidates requested from ranking before calibrating down to TOP_PAPERS_COUNT.
# Deliberately large: _reserve_exploration_slots (the heuristic path's own
# final cut) does no intermediate widening — whatever top_k it's given becomes
# the literal cut from the full ~100-candidate pool. A factor too small here
# means an under-represented category never reaches _calibrate_top_k at all
# because it was already sliced away one step earlier, silently defeating the
# whole point of calibrating. 8x (40 of ~100) was chosen after confirming 3x
# (15) reproducibly lost an entire category in a plausibly-skewed pool; it is
# still a heuristic window, not a guarantee — see the zero-representation log
# in _calibrate_top_k for when even this isn't enough.
CALIBRATION_OVERSAMPLE_FACTOR = 8

# A category's quota is only filled from candidates whose normalized relevance
# (0 = worst, 1 = best, in the oversampled pool) clears this floor. Without it,
# an empty or near-empty category would get filled by whatever's left in it —
# including a near-zero-relevance paper — ahead of a genuinely strong paper
# from an over-represented category. Tested empirically: below this floor,
# calibration starts trading a great paper for a barely-relevant one just to
# hit the target ratio, which is a worse result than the imbalance it fixes.
CALIBRATION_MIN_QUALITY = 0.15

def _blend_weights(novelty: float) -> Dict[str, float]:
    """
    Map a novelty preference in [0, 1] to ranking weights.

      novelty = 0.0  ->  {semantic .55, citation .45, recency .00}   established work
      novelty = 0.5  ->  {semantic .55, citation .23, recency .23}   mixed
      novelty = 1.0  ->  {semantic .55, citation .00, recency .45}   fresh work

    Continuous by construction, so a shift in reading habits nudges the ordering
    instead of swapping it wholesale.
    """
    novelty = min(1.0, max(0.0, novelty))
    return {
        "semantic": SEMANTIC_WEIGHT,
        "citation": TRADEOFF_BUDGET * (1.0 - novelty),
        "recency": TRADEOFF_BUDGET * novelty,
    }

def _describe_novelty(novelty: float) -> str:
    """Render the dial as a phrase for LLM prompts, which reason over prose."""
    if novelty >= 0.70:
        return "prefers recent, cutting-edge work over established literature"
    if novelty >= 0.55:
        return "leans toward recent work while still valuing established results"
    if novelty <= 0.30:
        return "prefers well-established, highly-cited literature"
    if novelty <= 0.45:
        return "leans toward established literature while open to recent work"
    return "balances established literature and recent work"

FOCUS_AREA_TERMS = {
    "ML": "machine learning",
    "NLP": "natural language processing",
    "CV": "computer vision",
    "AI": "artificial intelligence",
    "DL": "deep learning",
    "LLM": "large language models",
    "AGENT": "AI agents",
}

# Same keys as FOCUS_AREA_TERMS, mapped to the arXiv category codes actually
# indexed by run_nightly_index (settings.ARXIV_CATEGORIES) — used to give
# declared interests a prior in _compute_target_category_distribution. DL has
# no distinct arXiv code of its own, so it shares cs.LG with ML; LLM shares
# cs.CL with NLP for the same reason.
FOCUS_AREA_CATEGORY_MAP = {
    "ML": "cs.LG",
    "NLP": "cs.CL",
    "CV": "cs.CV",
    "AI": "cs.AI",
    "DL": "cs.LG",
    "LLM": "cs.CL",
    "AGENT": "cs.MA",
}

def _expand_focus_areas(areas: List[str]) -> List[str]:
    return [FOCUS_AREA_TERMS.get(a.upper(), a) for a in areas]
