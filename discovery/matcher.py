from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
import re
from typing import Iterable, List, Optional

from .config import MatchingConfig
from .models import GateFilteredPair, LowConfidencePair, MarketPair, OpinionMarketCandidate, PolymarketMarket
from .utils import normalize_text


@dataclass(slots=True)
class MatchStats:
    opinion_markets: int = 0
    polymarket_markets: int = 0
    matched: int = 0
    below_min_confidence: int = 0
    time_filtered: int = 0
    title_gate_filtered: int = 0


@dataclass(slots=True)
class _ScoredCandidate:
    opinion: OpinionMarketCandidate
    pm_market: PolymarketMarket
    score: float
    details: dict


def match_markets(
    opinion_markets: Iterable[OpinionMarketCandidate],
    polymarket_markets: Iterable[PolymarketMarket],
    config: MatchingConfig,
    logger=None,
    candidate_map: Optional[dict[str, List[PolymarketMarket]]] = None,
    event_title_map: Optional[dict[str, str]] = None,
) -> tuple[List[MarketPair], MatchStats, List[LowConfidencePair], List[GateFilteredPair]]:
    opinion_list = list(opinion_markets)
    polymarket_list = list(polymarket_markets)
    stats = MatchStats(
        opinion_markets=len(opinion_list),
        polymarket_markets=len(polymarket_list),
    )
    pm_prepared: dict[str, tuple[PolymarketMarket, dict]] = {}
    for market in polymarket_list:
        event_title = None
        if event_title_map:
            event_title = event_title_map.get(market.market_id)
        pm_prepared[market.market_id] = _prepare_polymarket(market, event_title=event_title)
    pm_index_all = list(pm_prepared.values())
    pairs: List[MarketPair] = []
    low_confidence: List[LowConfidencePair] = []
    gate_filtered: List[GateFilteredPair] = []
    all_edges: List[_ScoredCandidate] = []
    opinions_with_candidates: List[OpinionMarketCandidate] = []
    best_by_opinion: dict[str, _ScoredCandidate] = {}
    log_limit = 10
    logged = 0
    for opinion in opinion_list:
        parent_title = _opinion_event_title(opinion)
        parent_norm = normalize_text(parent_title)
        child_title = (opinion.title or "").strip()
        child_norm = normalize_text(child_title)
        child_gate_required = bool(child_norm) and child_norm != parent_norm
        best_parent_fail: tuple[
            float,
            PolymarketMarket,
            dict,
            Optional[float],
            List[str],
        ] | None = None
        best_child_fail: tuple[
            float,
            PolymarketMarket,
            dict,
            Optional[float],
            Optional[float],
            List[str],
        ] | None = None
        candidate_edges: List[_ScoredCandidate] = []
        pm_candidates = None
        if candidate_map is not None:
            pm_candidates = candidate_map.get(opinion.root_event_id)
        if pm_candidates is None:
            pm_index = pm_index_all
        else:
            pm_index = []
            for market in pm_candidates:
                prepared = pm_prepared.get(market.market_id)
                if prepared is None:
                    event_title = None
                    if event_title_map:
                        event_title = event_title_map.get(market.market_id)
                    prepared = _prepare_polymarket(market, event_title=event_title)
                    pm_prepared[market.market_id] = prepared
                pm_index.append(prepared)
        passed_parent_gate = False
        passed_gate = False
        for pm_market, pm_meta in pm_index:
            parent_gate_reasons: List[str] = []
            event_title_score = _event_title_similarity(parent_title, pm_meta, gate_reasons=parent_gate_reasons)
            event_score_value = event_title_score if event_title_score is not None else 0.0
            if event_title_score is None or event_title_score < config.title_gate_min:
                if best_parent_fail is None or event_score_value > best_parent_fail[0]:
                    best_parent_fail = (
                        event_score_value,
                        pm_market,
                        pm_meta,
                        event_title_score,
                        list(parent_gate_reasons),
                    )
                stats.title_gate_filtered += 1
                continue
            passed_parent_gate = True
            child_title_score = None
            if child_gate_required:
                pm_child_title = pm_market.group_item_title or pm_market.question or ""
                child_gate_reasons: List[str] = []
                pm_parent_title = pm_meta.get("event_title_raw") or pm_market.event_title or pm_market.question or ""
                child_title_score = _child_gate_score(
                    child_title,
                    pm_child_title,
                    opinion_parent_title=parent_title,
                    pm_parent_title=pm_parent_title,
                    gate_reasons=child_gate_reasons,
                )
                child_score_value = child_title_score if child_title_score is not None else 0.0
                if child_title_score is None or child_title_score < config.child_title_gate_min:
                    if best_child_fail is None or child_score_value > best_child_fail[0]:
                        best_child_fail = (
                            child_score_value,
                            pm_market,
                            pm_meta,
                            event_title_score,
                            child_title_score,
                            list(child_gate_reasons),
                        )
                    stats.title_gate_filtered += 1
                    continue
            passed_gate = True
            if not _within_time_window(opinion, pm_market, config):
                continue
            score, details = _score_match(
                opinion,
                pm_market,
                pm_meta,
                config,
                title_score=1.0,
                event_title_score=event_title_score,
                child_title_score=child_title_score,
            )
            candidate = _ScoredCandidate(
                opinion=opinion,
                pm_market=pm_market,
                score=score,
                details=details,
            )
            candidate_edges.append(candidate)
        if not candidate_edges:
            if passed_gate:
                stats.time_filtered += 1
            elif passed_parent_gate and best_child_fail is not None:
                gate_filtered.append(
                    _build_gate_filtered_pair(
                        opinion=opinion,
                        pm_market=best_child_fail[1],
                        pm_meta=best_child_fail[2],
                        gate_stage="child",
                        gate_score=best_child_fail[0],
                        gate_threshold=config.child_title_gate_min,
                        event_title_score=best_child_fail[3],
                        child_title_score=best_child_fail[4],
                        gate_reasons=best_child_fail[5],
                        event_title_map=event_title_map,
                    )
                )
            elif not passed_parent_gate and best_parent_fail is not None:
                gate_filtered.append(
                    _build_gate_filtered_pair(
                        opinion=opinion,
                        pm_market=best_parent_fail[1],
                        pm_meta=best_parent_fail[2],
                        gate_stage="parent",
                        gate_score=best_parent_fail[0],
                        gate_threshold=config.title_gate_min,
                        event_title_score=best_parent_fail[3],
                        child_title_score=None,
                        gate_reasons=best_parent_fail[4],
                        event_title_map=event_title_map,
                    )
                )
            continue
        best_candidate = max(candidate_edges, key=lambda item: item.score)
        best_by_opinion[opinion.market_id] = best_candidate
        opinions_with_candidates.append(opinion)
        all_edges.extend(candidate_edges)
    all_edges.sort(key=lambda item: item.score, reverse=True)
    assigned: dict[str, _ScoredCandidate] = {}
    used_pm: set[str] = set()
    for edge in all_edges:
        op_id = edge.opinion.market_id
        pm_id = edge.pm_market.market_id
        if op_id in assigned or pm_id in used_pm:
            continue
        assigned[op_id] = edge
        used_pm.add(pm_id)
    for opinion in opinions_with_candidates:
        edge = assigned.get(opinion.market_id)
        if edge is None:
            best_candidate = best_by_opinion.get(opinion.market_id)
            if best_candidate is not None:
                gate_filtered.append(
                    _build_assignment_unmatched_pair(
                        opinion=opinion,
                        best_score=best_candidate.score,
                        best_details=best_candidate.details,
                        min_confidence=config.min_confidence,
                    )
                )
            continue
        if edge.score < config.min_confidence:
            stats.below_min_confidence += 1
            low_confidence.append(
                _build_low_confidence_pair(
                    opinion,
                    edge.pm_market,
                    edge.score,
                    edge.details or {},
                    event_title_map,
                )
            )
            if logger and logged < log_limit:
                pm_url = _polymarket_url(edge.pm_market)
                event_title = _opinion_event_title(opinion)
                child_title = opinion.title or ""
                logger.info(
                    "match below_min op_market_id=%s op_event_title=%s op_child_title=%s score=%.4f min=%.4f event_title=%.4f title=%.4f labels=%.4f teams=%.4f time=%.4f pm_market_id=%s pm_name=%s pm_url=%s",
                    opinion.market_id,
                    _log_title(event_title),
                    _log_title(child_title),
                    edge.score,
                    config.min_confidence,
                    _score_value(edge.details, "event_title"),
                    _score_value(edge.details, "title"),
                    _score_value(edge.details, "labels"),
                    _score_value(edge.details, "teams"),
                    _score_value(edge.details, "time"),
                    edge.pm_market.market_id,
                    edge.pm_market.question,
                    pm_url,
                )
                logged += 1
            continue
        pair = _build_pair(opinion, edge.pm_market, edge.score, edge.details or {}, event_title_map)
        pairs.append(pair)
        stats.matched += 1
    if logger:
        logger.info(
            "match summary opinion=%d polymarket=%d matched=%d below_min=%d time_filtered=%d title_gate_filtered=%d",
            stats.opinion_markets,
            stats.polymarket_markets,
            stats.matched,
            stats.below_min_confidence,
            stats.time_filtered,
            stats.title_gate_filtered,
        )
    return pairs, stats, low_confidence, gate_filtered


def _prepare_polymarket(
    market: PolymarketMarket,
    event_title: Optional[str] = None,
) -> tuple[PolymarketMarket, dict]:
    title_norm = normalize_text(market.question or "")
    event_title_value = event_title or market.event_title or ""
    event_title_raw = event_title_value.strip() if event_title_value else ""
    event_title_norm = normalize_text(event_title_raw) if event_title_raw else ""
    title_raw = (market.question or "").strip()
    teams_norm = _normalize_teams(market.team_a, market.team_b)
    outcomes_norm = _normalize_outcomes(market.outcomes)
    return market, {
        "title_norm": title_norm,
        "event_title_norm": event_title_norm,
        "title_raw": title_raw,
        "event_title_raw": event_title_raw,
        "teams_norm": teams_norm,
        "outcomes_norm": outcomes_norm,
    }


def _within_time_window(
    opinion: OpinionMarketCandidate,
    pm_market: PolymarketMarket,
    config: MatchingConfig,
) -> bool:
    max_hours = config.max_time_diff_hours
    if max_hours <= 0:
        return True
    if opinion.start_time and pm_market.start_time:
        diff = abs((opinion.start_time - pm_market.start_time).total_seconds()) / 3600
        return diff <= max_hours
    if opinion.end_time and pm_market.end_time:
        diff = abs((opinion.end_time - pm_market.end_time).total_seconds()) / 3600
        return diff <= max_hours
    return True


def _score_match(
    opinion: OpinionMarketCandidate,
    pm_market: PolymarketMarket,
    pm_meta: dict,
    config: MatchingConfig,
    title_score: Optional[float],
    event_title_score: Optional[float],
    child_title_score: Optional[float],
) -> tuple[float, dict]:
    teams_score = _teams_similarity(opinion.team_a, opinion.team_b, pm_meta.get("teams_norm"))
    labels_score = _labels_similarity(opinion, pm_meta.get("outcomes_norm"))
    numeric_score = _numeric_range_similarity(opinion.title or "", pm_market.question or "")
    if numeric_score is not None:
        if labels_score is None or numeric_score > labels_score:
            labels_score = numeric_score
    time_score = _time_similarity(opinion.start_time, opinion.end_time, pm_market.start_time, pm_market.end_time, config)
    weights = {
        "title": config.weight_title if title_score is not None else 0.0,
        "teams": config.weight_teams if teams_score is not None else 0.0,
        "labels": config.weight_labels if labels_score is not None else 0.0,
        "time": config.weight_time if time_score is not None else 0.0,
    }
    total_weight = sum(weights.values())
    if total_weight <= 0:
        return 0.0, {
            "event_title": event_title_score,
            "title": title_score,
            "teams": teams_score,
            "labels": labels_score,
            "time": time_score,
        }
    score = 0.0
    if title_score is not None:
        score += title_score * weights["title"]
    if teams_score is not None:
        score += teams_score * weights["teams"]
    if labels_score is not None:
        score += labels_score * weights["labels"]
    if time_score is not None:
        score += time_score * weights["time"]
    score = score / total_weight
    return score, {
        "event_title": event_title_score,
        "title": title_score,
        "title_child": child_title_score,
        "teams": teams_score,
        "labels": labels_score,
        "numeric_range": numeric_score,
        "time": time_score,
    }


def _event_title_similarity(
    opinion_title: str,
    pm_meta: dict,
    gate_reasons: Optional[List[str]] = None,
) -> Optional[float]:
    if not opinion_title:
        return None
    event_title_raw = pm_meta.get("event_title_raw") or ""
    title_raw = pm_meta.get("title_raw") or ""
    candidate_title = event_title_raw or title_raw
    if candidate_title:
        date_score = _date_gate_score(opinion_title, candidate_title)
        if date_score == 0.0:
            _add_gate_reason(gate_reasons, "parent_date_mismatch")
            return 0.0
        left_sig = _intent_signature(opinion_title)
        right_sig = _intent_signature(candidate_title)
        if left_sig["question_type"] or right_sig["question_type"]:
            if left_sig["question_type"] != right_sig["question_type"]:
                _add_gate_reason(gate_reasons, "parent_question_type_mismatch")
                return 0.0
        if left_sig["comparison"] != right_sig["comparison"]:
            if left_sig["comparison"] or right_sig["comparison"]:
                _add_gate_reason(gate_reasons, "parent_comparison_mismatch")
                return 0.0
        if left_sig["category_tokens"] or right_sig["category_tokens"]:
            if not left_sig["category_tokens"] or not right_sig["category_tokens"]:
                _add_gate_reason(gate_reasons, "parent_category_mismatch")
                return 0.0
            if left_sig["category_tokens"].isdisjoint(right_sig["category_tokens"]):
                _add_gate_reason(gate_reasons, "parent_category_mismatch")
                return 0.0
        if left_sig["proper_nouns"] or right_sig["proper_nouns"]:
            if not left_sig["proper_nouns"] or not right_sig["proper_nouns"]:
                _add_gate_reason(gate_reasons, "parent_proper_noun_mismatch")
                return 0.0
            if left_sig["proper_nouns"].isdisjoint(right_sig["proper_nouns"]):
                _add_gate_reason(gate_reasons, "parent_proper_noun_mismatch")
                return 0.0
        if left_sig["outcome_groups"] or right_sig["outcome_groups"]:
            if not left_sig["outcome_groups"] or not right_sig["outcome_groups"]:
                _add_gate_reason(gate_reasons, "parent_outcome_mismatch")
                return 0.0
            if left_sig["outcome_groups"].isdisjoint(right_sig["outcome_groups"]):
                _add_gate_reason(gate_reasons, "parent_outcome_mismatch")
                return 0.0
        if left_sig["action_groups"] or right_sig["action_groups"]:
            if not left_sig["action_groups"] or not right_sig["action_groups"]:
                _add_gate_reason(gate_reasons, "parent_action_mismatch")
                return 0.0
            if left_sig["action_groups"].isdisjoint(right_sig["action_groups"]):
                _add_gate_reason(gate_reasons, "parent_action_mismatch")
                return 0.0
        priority_op = left_sig["priority_entities"]
        priority_pm = right_sig["priority_entities"]
        allow_single = False
        if priority_op and priority_pm:
            if priority_op.isdisjoint(priority_pm):
                _add_gate_reason(gate_reasons, "parent_priority_entity_mismatch")
                return 0.0
            allow_single = True
        op_entities = left_sig["entity_tokens"]
        pm_entities = right_sig["entity_tokens"]
        if op_entities or pm_entities:
            if not op_entities or not pm_entities:
                _add_gate_reason(gate_reasons, "parent_entity_mismatch")
                return 0.0
            score = _entity_match_score(op_entities, pm_entities, allow_single=allow_single)
            if score <= 0.0:
                _add_gate_reason(gate_reasons, "parent_entity_mismatch")
                return 0.0
            return score
    event_title_norm = pm_meta.get("event_title_norm") or ""
    if event_title_norm:
        return _title_similarity(opinion_title, event_title_norm)
    title_norm = pm_meta.get("title_norm") or ""
    if title_norm:
        return _title_similarity(opinion_title, title_norm)
    return None


_ENTITY_STOPWORDS = {
    "a",
    "an",
    "any",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "in",
    "into",
    "is",
    "it",
    "its",
    "may",
    "might",
    "more",
    "most",
    "must",
    "no",
    "not",
    "of",
    "on",
    "or",
    "out",
    "over",
    "should",
    "some",
    "than",
    "that",
    "the",
    "their",
    "there",
    "these",
    "this",
    "those",
    "to",
    "under",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "will",
    "with",
    "without",
    "would",
    "above",
    "after",
    "before",
    "below",
    "between",
    "by",
    "end",
    "start",
    "closing",
    "close",
    "market",
    "price",
    "cap",
    "fdv",
    "launch",
    "token",
    "tokens",
    "ipo",
    "visit",
    "leave",
    "leaves",
    "left",
    "country",
    "countries",
    "company",
    "companies",
    "win",
    "wins",
    "won",
    "lose",
    "loses",
    "lost",
    "beat",
    "beats",
    "decision",
    "rate",
    "rates",
    "increase",
    "decrease",
    "change",
    "best",
    "winner",
    "winners",
    "above",
    "below",
    "over",
    "under",
    "after",
    "before",
    "between",
    "at",
    "least",
    "most",
    "more",
    "less",
}

_WEAK_ENTITIES = {
    "us",
    "usa",
    "nba",
    "bitcoin",
    "btc",
    "champion",
    "champions",
    "countries",
    "country",
    "decision",
    "ipo",
    "launch",
    "leave",
    "rate",
    "rates",
    "token",
    "tokens",
    "visit",
    "winner",
    "winners",
}

_MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

_COMPARISON_TOKENS = {
    "before",
    "after",
    "between",
    "vs",
    "versus",
    "x",
}

_ACTION_GROUPS = {
    "invade": {
        "invade",
        "invades",
        "invaded",
        "invasion",
        "occupy",
        "occupies",
        "occupied",
        "occupation",
        "annex",
        "annexed",
        "annexation",
    },
    "attack": {
        "attack",
        "attacks",
        "attacked",
        "assault",
        "assaults",
        "strike",
        "strikes",
        "struck",
    },
    "clash": {
        "clash",
        "clashes",
        "skirmish",
        "skirmishes",
        "conflict",
        "war",
        "battle",
        "fighting",
        "fight",
    },
    "test": {
        "test",
        "tests",
        "testing",
        "nuclear",
        "missile",
        "launch",
        "launched",
        "detonation",
        "explosion",
    },
}

_OUTCOME_GROUPS = {
    "winner": {
        "winner",
        "winners",
        "champion",
        "champions",
        "win",
        "wins",
        "won",
        "title",
    },
    "qualify": {
        "qualify",
        "qualified",
        "qualifies",
        "qualifying",
        "qualification",
        "advance",
        "advances",
        "advanced",
    },
}

_DIRECTION_UP_WORDS = {
    "increase",
    "increases",
    "increased",
    "raise",
    "raises",
    "raised",
    "hike",
    "hikes",
    "hiked",
    "up",
}

_DIRECTION_DOWN_WORDS = {
    "decrease",
    "decreases",
    "decreased",
    "cut",
    "cuts",
    "cutting",
    "lower",
    "lowers",
    "lowered",
    "down",
}

_CHILD_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "of",
    "to",
    "for",
    "by",
    "in",
    "on",
    "at",
    "vs",
    "versus",
    "party",
    "election",
    "winner",
    "winners",
    "season",
    "year",
    "gaming",
    "esports",
    "team",
    "club",
    "fc",
    "cf",
    "sc",
    "afc",
}

_CHILD_CATEGORICAL_MIN = 0.9

_CATEGORY_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "of",
    "to",
    "for",
    "by",
    "in",
    "on",
    "at",
    "vs",
    "versus",
    "winner",
    "winners",
    "best",
    "award",
    "awards",
    "season",
    "year",
    "champion",
    "champions",
    "league",
    "election",
    "mayoral",
}

_PROPER_NOUN_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "of",
    "to",
    "for",
    "by",
    "in",
    "on",
    "at",
    "vs",
    "versus",
    "best",
    "winner",
    "winners",
    "season",
    "year",
    "award",
    "awards",
    "champion",
    "champions",
    "election",
    "mayoral",
    "party",
    "bank",
    "decision",
    "rate",
    "rates",
}
_MONTH_PATTERN = (
    r"jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:tember)?|sept(?:ember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?"
)
_DATE_MONTH_DAY_YEAR_RE = re.compile(
    rf"\b(?P<month>{_MONTH_PATTERN})\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?(?:,|\s)+(?P<year>\d{{4}})\b",
    flags=re.IGNORECASE,
)
_DATE_DAY_MONTH_YEAR_RE = re.compile(
    rf"\b(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s+(?P<month>{_MONTH_PATTERN})\s+(?P<year>\d{{4}})\b",
    flags=re.IGNORECASE,
)
_DATE_MONTH_DAY_RE = re.compile(
    rf"\b(?P<month>{_MONTH_PATTERN})\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\b(?!\s*,?\s*\d{{4}})",
    flags=re.IGNORECASE,
)
_DATE_DAY_MONTH_RE = re.compile(
    rf"\b(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s+(?P<month>{_MONTH_PATTERN})\b(?!\s*\d{{4}})",
    flags=re.IGNORECASE,
)
_DATE_YEAR_FIRST_RE = re.compile(
    r"\b(?P<year>\d{4})[/-](?P<month>\d{1,2})[/-](?P<day>\d{1,2})\b"
)
_DATE_MONTH_FIRST_RE = re.compile(
    r"\b(?P<month>\d{1,2})[/-](?P<day>\d{1,2})[/-](?P<year>\d{4})\b"
)
_DATE_MONTH_DAY_NUM_RE = re.compile(
    r"\b(?P<month>\d{1,2})[/-](?P<day>\d{1,2})\b(?![/-]\d{2,4})"
)
_DATE_MONTH_YEAR_RE = re.compile(
    rf"\b(?P<month>{_MONTH_PATTERN})\s+(?P<year>\d{{4}})\b",
    flags=re.IGNORECASE,
)
_DATE_YEAR_MONTH_RE = re.compile(
    rf"\b(?P<year>\d{{4}})\s+(?P<month>{_MONTH_PATTERN})\b",
    flags=re.IGNORECASE,
)
_DATE_MONTH_YEAR_NUM_RE = re.compile(
    r"\b(?P<month>\d{1,2})[/-](?P<year>\d{4})\b(?![/-]\d{1,2})"
)
_DATE_YEAR_MONTH_NUM_RE = re.compile(
    r"\b(?P<year>\d{4})[/-](?P<month>\d{1,2})\b(?![/-]\d{1,2})"
)
_DATE_MONTH_ONLY_RE = re.compile(
    rf"\b(?P<month>{_MONTH_PATTERN})\b",
    flags=re.IGNORECASE,
)
_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2}|21\d{2})\b")


def _entity_tokens(text: str) -> set[str]:
    text = re.sub(r"\bu\\.?\\s*s\\.?\b", "us", text or "", flags=re.IGNORECASE)
    normalized = normalize_text(text)
    tokens: set[str] = set()
    for token in normalized.split():
        if not token:
            continue
        if len(token) <= 1:
            continue
        if token in _WEAK_ENTITIES:
            tokens.add(token)
            continue
        if token in _ENTITY_STOPWORDS:
            continue
        if token in _MONTHS:
            continue
        if re.search(r"\d", token):
            continue
        tokens.add(token)
    return tokens


def _entity_overlap_score(left: set[str], right: set[str]) -> float:
    intersection = left.intersection(right)
    if not intersection:
        return 0.0
    return len(intersection) / max(len(left), len(right))


def _entity_match_score(left: set[str], right: set[str], allow_single: bool = False) -> float:
    overlap = left.intersection(right)
    if not overlap:
        return 0.0
    strong_left = left.difference(_WEAK_ENTITIES)
    strong_right = right.difference(_WEAK_ENTITIES)
    strong_overlap = strong_left.intersection(strong_right)
    if allow_single and strong_overlap:
        return _entity_overlap_score(left, right)
    if len(strong_overlap) >= 2:
        return _entity_overlap_score(left, right)
    if len(strong_overlap) >= 1 and len(overlap) >= 2:
        return _entity_overlap_score(left, right)
    return 0.0


_PAREN_CONTENT_RE = re.compile(r"\(([^)]+)\)")
_TICKER_RE = re.compile(r"\b[A-Z0-9]{2,6}\b")
_PRIORITY_TOKEN_RE = re.compile(r"\b[A-Za-z0-9][A-Za-z0-9-]*\b")


def _priority_entities(text: str) -> set[str]:
    if not text:
        return set()
    tokens: set[str] = set()
    for match in _PAREN_CONTENT_RE.finditer(text):
        content = match.group(1)
        for token in _TICKER_RE.findall(content):
            if token.isdigit():
                continue
            token_norm = token.lower()
            if token_norm in _ENTITY_STOPWORDS or token_norm in _WEAK_ENTITIES:
                continue
            tokens.add(token_norm)
    for match in _PRIORITY_TOKEN_RE.finditer(text):
        token = match.group(0)
        if len(token) <= 1 or token.isdigit():
            continue
        token_norm = token.lower()
        if token_norm in _ENTITY_STOPWORDS or token_norm in _WEAK_ENTITIES:
            continue
        if _is_camel_token(token) or _is_alnum_token(token) or _is_caps_token(token):
            tokens.add(token_norm)
    return tokens


def _is_camel_token(token: str) -> bool:
    if not token.isalpha():
        return False
    if token.islower() or token.isupper():
        return False
    return bool(re.search(r"[a-z][A-Z]", token) or re.search(r"[A-Z].*[A-Z]", token))


def _is_alnum_token(token: str) -> bool:
    return re.search(r"[A-Za-z]", token) and re.search(r"\d", token)


def _is_caps_token(token: str) -> bool:
    return token.isupper() and 2 <= len(token) <= 6


def _add_gate_reason(reasons: Optional[List[str]], reason: str) -> None:
    if reasons is None:
        return
    if reason not in reasons:
        reasons.append(reason)


def _intent_signature(text: str) -> dict:
    return {
        "question_type": _question_type(text),
        "comparison": _has_comparison_term(text),
        "outcome_groups": _outcome_groups(text),
        "action_groups": _action_groups(text),
        "priority_entities": _priority_entities(text),
        "entity_tokens": _entity_tokens(text),
        "category_tokens": _category_tokens(text),
        "proper_nouns": _proper_noun_tokens(text),
    }


def _question_type(text: str) -> Optional[str]:
    if not text:
        return None
    normalized = normalize_text(text)
    if "how long" in normalized:
        return "how_long"
    if "how many" in normalized:
        return "how_many"
    if "what price" in normalized:
        return "what_price"
    if "what time" in normalized:
        return "what_time"
    return None


def _has_comparison_term(text: str) -> bool:
    if not text:
        return False
    normalized = normalize_text(text)
    tokens = set(normalized.split())
    return any(token in _COMPARISON_TOKENS for token in tokens)


def _action_groups(text: str) -> set[str]:
    if not text:
        return set()
    normalized = normalize_text(text)
    tokens = set(normalized.split())
    groups: set[str] = set()
    for name, keywords in _ACTION_GROUPS.items():
        if tokens.intersection(keywords):
            groups.add(name)
    return groups


def _outcome_groups(text: str) -> set[str]:
    if not text:
        return set()
    normalized = normalize_text(text)
    tokens = set(normalized.split())
    groups: set[str] = set()
    for name, keywords in _OUTCOME_GROUPS.items():
        if tokens.intersection(keywords):
            groups.add(name)
    return groups


def _direction_hint(text: str) -> Optional[str]:
    if not text:
        return None
    if "↑" in text:
        return "up"
    if "↓" in text:
        return "down"
    normalized = normalize_text(text)
    tokens = set(normalized.split())
    if tokens.intersection(_DIRECTION_UP_WORDS):
        return "up"
    if tokens.intersection(_DIRECTION_DOWN_WORDS):
        return "down"
    return None


def _categorical_tokens(text: str) -> List[str]:
    normalized = normalize_text(text)
    if not normalized:
        return []
    return [token for token in normalized.split() if token and token not in _CHILD_STOPWORDS]


def _acronym(tokens: List[str]) -> str:
    if len(tokens) < 2:
        return ""
    return "".join(token[0] for token in tokens if token)


def _tokens_overlap_or_prefix(left: List[str], right: List[str]) -> bool:
    left_set = set(left)
    right_set = set(right)
    if left_set.intersection(right_set):
        return True
    for lt in left:
        for rt in right:
            if len(lt) >= 4 and len(rt) >= 4:
                if lt.startswith(rt) or rt.startswith(lt):
                    return True
    return False


def _categorical_child_score(
    left_text: str,
    right_text: str,
    exclude_tokens: Optional[set[str]] = None,
    gate_reasons: Optional[List[str]] = None,
) -> Optional[float]:
    left_acros = _child_acronyms(left_text)
    right_acros = _child_acronyms(right_text)
    if left_acros and right_acros and left_acros.isdisjoint(right_acros):
        _add_gate_reason(gate_reasons, "child_acronym_mismatch")
        return 0.0
    left_tokens = _categorical_tokens(left_text)
    right_tokens = _categorical_tokens(right_text)
    if exclude_tokens:
        filtered_left = [token for token in left_tokens if token not in exclude_tokens]
        filtered_right = [token for token in right_tokens if token not in exclude_tokens]
        if filtered_left:
            left_tokens = filtered_left
        if filtered_right:
            right_tokens = filtered_right
    if not left_tokens or not right_tokens:
        return None
    if _tokens_overlap_or_prefix(left_tokens, right_tokens):
        return 1.0
    left_acr = _acronym(left_tokens)
    right_acr = _acronym(right_tokens)
    if left_acr and right_acr and left_acr == right_acr:
        return 1.0
    if left_acr and left_acr in right_tokens:
        return 1.0
    if right_acr and right_acr in left_tokens:
        return 1.0
    ratio = SequenceMatcher(None, normalize_text(left_text), normalize_text(right_text)).ratio()
    if ratio >= _CHILD_CATEGORICAL_MIN:
        return ratio
    _add_gate_reason(gate_reasons, "child_categorical_mismatch")
    return 0.0


def _category_tokens(text: str) -> set[str]:
    if not text:
        return set()
    raw = text
    if ":" in raw:
        raw = raw.split(":", 1)[1]
    normalized = normalize_text(raw)
    tokens: set[str] = set()
    for token in normalized.split():
        if not token or token.isdigit():
            continue
        if token in _CATEGORY_STOPWORDS or token in _MONTHS:
            continue
        tokens.add(token)
    return tokens


_PROPER_NOUN_RE = re.compile(r"[A-Z][A-Za-z]+|[A-Z]{2,}")

_CHILD_ACRONYM_RE = re.compile(r"\b[A-Z]{2,5}\b")


def _proper_noun_tokens(text: str) -> set[str]:
    if not text:
        return set()
    tokens: set[str] = set()
    for token in _PROPER_NOUN_RE.findall(text):
        token_norm = token.lower()
        if token_norm.isdigit():
            continue
        if token_norm in _PROPER_NOUN_STOPWORDS or token_norm in _MONTHS:
            continue
        tokens.add(token_norm)
    return tokens


def _child_acronyms(text: str) -> set[str]:
    if not text:
        return set()
    tokens: set[str] = set()
    for token in _CHILD_ACRONYM_RE.findall(text):
        token_norm = token.lower()
        if token_norm in _CHILD_STOPWORDS:
            continue
        tokens.add(token_norm)
    return tokens


def _normalize_date_text(text: str) -> str:
    text = (text or "").translate(_DASH_TRANSLATION)
    text = text.lower()
    text = re.sub(r"[^0-9a-z/\\s-]", " ", text)
    return " ".join(text.split())


def _extract_date_ordinals(text: str) -> List[int]:
    if not text:
        return []
    normalized = _normalize_date_text(text)
    ordinals: set[int] = set()
    for match in _DATE_MONTH_DAY_YEAR_RE.finditer(normalized):
        month = _MONTHS.get(match.group("month"))
        day = int(match.group("day"))
        year = int(match.group("year"))
        if month is None:
            continue
        try:
            ordinals.add(datetime(year, month, day).date().toordinal())
        except ValueError:
            continue
    for match in _DATE_DAY_MONTH_YEAR_RE.finditer(normalized):
        month = _MONTHS.get(match.group("month"))
        day = int(match.group("day"))
        year = int(match.group("year"))
        if month is None:
            continue
        try:
            ordinals.add(datetime(year, month, day).date().toordinal())
        except ValueError:
            continue
    for match in _DATE_YEAR_FIRST_RE.finditer(normalized):
        year = int(match.group("year"))
        month = int(match.group("month"))
        day = int(match.group("day"))
        try:
            ordinals.add(datetime(year, month, day).date().toordinal())
        except ValueError:
            continue
    for match in _DATE_MONTH_FIRST_RE.finditer(normalized):
        month = int(match.group("month"))
        day = int(match.group("day"))
        year = int(match.group("year"))
        try:
            ordinals.add(datetime(year, month, day).date().toordinal())
        except ValueError:
            continue
    return sorted(ordinals)


def _extract_years(text: str) -> set[int]:
    normalized = _normalize_date_text(text)
    years: set[int] = set()
    for match in _YEAR_RE.finditer(normalized):
        years.add(int(match.group(1)))
    return years


def _extract_month_days(text: str) -> set[tuple[int, int]]:
    if not text:
        return set()
    normalized = _normalize_date_text(text)
    month_days: set[tuple[int, int]] = set()
    for match in _DATE_MONTH_DAY_RE.finditer(normalized):
        month = _MONTHS.get(match.group("month"))
        day = int(match.group("day"))
        if month is None or not (1 <= day <= 31):
            continue
        month_days.add((month, day))
    for match in _DATE_DAY_MONTH_RE.finditer(normalized):
        month = _MONTHS.get(match.group("month"))
        day = int(match.group("day"))
        if month is None or not (1 <= day <= 31):
            continue
        month_days.add((month, day))
    for match in _DATE_MONTH_DAY_NUM_RE.finditer(normalized):
        month = int(match.group("month"))
        day = int(match.group("day"))
        if 1 <= month <= 12 and 1 <= day <= 31:
            month_days.add((month, day))
    return month_days


def _extract_month_years(text: str) -> set[tuple[int, int]]:
    if not text:
        return set()
    normalized = _normalize_date_text(text)
    months: set[tuple[int, int]] = set()
    for match in _DATE_MONTH_YEAR_RE.finditer(normalized):
        month = _MONTHS.get(match.group("month"))
        year = int(match.group("year"))
        if month is None:
            continue
        months.add((year, month))
    for match in _DATE_YEAR_MONTH_RE.finditer(normalized):
        month = _MONTHS.get(match.group("month"))
        year = int(match.group("year"))
        if month is None:
            continue
        months.add((year, month))
    for match in _DATE_MONTH_YEAR_NUM_RE.finditer(normalized):
        month = int(match.group("month"))
        year = int(match.group("year"))
        if 1 <= month <= 12:
            months.add((year, month))
    for match in _DATE_YEAR_MONTH_NUM_RE.finditer(normalized):
        month = int(match.group("month"))
        year = int(match.group("year"))
        if 1 <= month <= 12:
            months.add((year, month))
    return months


def _extract_months(text: str) -> set[int]:
    if not text:
        return set()
    normalized = _normalize_date_text(text)
    months: set[int] = set()
    for match in _DATE_MONTH_ONLY_RE.finditer(normalized):
        month = _MONTHS.get(match.group("month"))
        if month is None:
            continue
        months.add(month)
    return months


def _date_month_pairs(ordinals: Iterable[int]) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for ordinal in ordinals:
        try:
            date_value = datetime.fromordinal(ordinal).date()
        except (OverflowError, ValueError):
            continue
        pairs.add((date_value.year, date_value.month))
    return pairs


def _date_month_day_pairs(ordinals: Iterable[int]) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for ordinal in ordinals:
        try:
            date_value = datetime.fromordinal(ordinal).date()
        except (OverflowError, ValueError):
            continue
        pairs.add((date_value.month, date_value.day))
    return pairs


def _date_gate_score(left: str, right: str) -> Optional[float]:
    left_dates = _extract_date_ordinals(left)
    right_dates = _extract_date_ordinals(right)
    left_month_days = _extract_month_days(left)
    right_month_days = _extract_month_days(right)
    left_month_years = _extract_month_years(left)
    right_month_years = _extract_month_years(right)
    if left_dates or right_dates:
        if not left_dates or not right_dates:
            if left_dates and right_month_days:
                if _date_month_day_pairs(left_dates).intersection(right_month_days):
                    return 1.0
                return 0.0
            if right_dates and left_month_days:
                if _date_month_day_pairs(right_dates).intersection(left_month_days):
                    return 1.0
                return 0.0
            return 0.0
        if set(left_dates).intersection(right_dates):
            return 1.0
        return 0.0
    if left_month_days or right_month_days:
        if not left_month_days or not right_month_days:
            return 0.0
        if left_month_days.intersection(right_month_days):
            return 1.0
        return 0.0
    if left_month_years or right_month_years:
        if not left_month_years or not right_month_years:
            return 0.0
        if left_month_years.intersection(right_month_years):
            return 1.0
        return 0.0
    left_months = _extract_months(left)
    right_months = _extract_months(right)
    if left_months or right_months:
        if not left_months or not right_months:
            return 0.0
        if left_months.intersection(right_months):
            return 1.0
        return 0.0
    left_years = _extract_years(left)
    right_years = _extract_years(right)
    if left_years or right_years:
        if not left_years or not right_years:
            return 0.0
        if left_years.intersection(right_years):
            return 1.0
        return 0.0
    return None


def _child_gate_score(
    child_title: str,
    pm_question: str,
    opinion_parent_title: str | None = None,
    pm_parent_title: str | None = None,
    gate_reasons: Optional[List[str]] = None,
) -> Optional[float]:
    if not child_title or not pm_question:
        return None
    child_norm = normalize_text(child_title)
    if not child_norm:
        return None
    question_norm = normalize_text(pm_question)
    if not question_norm:
        return None
    date_score = _date_gate_score(child_title, pm_question)
    if date_score is not None:
        if date_score == 0.0:
            _add_gate_reason(gate_reasons, "child_date_mismatch")
        return date_score
    left_direction = _direction_hint(child_title)
    right_direction = _direction_hint(pm_question)
    if left_direction and right_direction and left_direction != right_direction:
        _add_gate_reason(gate_reasons, "child_direction_mismatch")
        return 0.0
    left_numeric = _has_numeric_signal(child_title)
    right_numeric = _has_numeric_signal(pm_question)
    if left_numeric != right_numeric:
        _add_gate_reason(gate_reasons, "child_type_mismatch")
        return 0.0
    if left_numeric and right_numeric:
        left_bounds = _parse_numeric_bounds(child_title)
        right_bounds = _parse_numeric_bounds(pm_question)
        if not left_bounds or not right_bounds:
            _add_gate_reason(gate_reasons, "child_numeric_parse_failed")
            return 0.0
        if left_bounds.kind != right_bounds.kind:
            _add_gate_reason(gate_reasons, "child_numeric_kind_mismatch")
            return 0.0
        score = _bounds_similarity(
            (left_bounds.low, left_bounds.high),
            (right_bounds.low, right_bounds.high),
        )
        if score == 0.0:
            _add_gate_reason(gate_reasons, "child_numeric_nonoverlap")
        return score
    if _contains_token(child_norm, question_norm):
        return 1.0
    parent_tokens = _categorical_tokens(opinion_parent_title or "")
    parent_tokens.extend(_categorical_tokens(pm_parent_title or ""))
    exclude_tokens = set(parent_tokens)
    categorical_score = _categorical_child_score(
        child_title,
        pm_question,
        exclude_tokens=exclude_tokens,
        gate_reasons=gate_reasons,
    )
    if categorical_score is not None:
        return categorical_score
    return _title_similarity(child_title, question_norm)


def _contains_token(needle: str, haystack: str) -> bool:
    needle_tokens = needle.split()
    haystack_tokens = set(haystack.split())
    return bool(needle_tokens) and all(token in haystack_tokens for token in needle_tokens)

@dataclass(frozen=True, slots=True)
class _NumericToken:
    raw_value: float
    multiplier: float
    kind: str
    has_unit: bool


@dataclass(frozen=True, slots=True)
class _NumericBounds:
    low: Optional[float]
    high: Optional[float]
    kind: str


_DASH_TRANSLATION = str.maketrans(
    {
        "–": "-",
        "—": "-",
        "−": "-",
        "‑": "-",
        "‒": "-",
        "―": "-",
    }
)

_NUMERIC_TOKEN_PATTERN = (
    r"[$€£]?\s*\d[\d,]*(?:\.\d+)?\s*"
    r"(?:t|trillion|tn|b|bn|billion|m|mm|mn|million|k|thousand|%|percent|pct|bps|bp|basis\s*points?)?"
)
_NUMERIC_VALUE_RE = re.compile(
    r"(?P<prefix>[$€£])?\s*(?P<num>\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<unit>t|trillion|tn|b|bn|billion|m|mm|mn|million|k|thousand|%|percent|pct|bps|bp|basis\s*points?)?",
    flags=re.IGNORECASE,
)
_NUMERIC_BETWEEN_RE = re.compile(
    rf"\bbetween\s+(?P<first>{_NUMERIC_TOKEN_PATTERN})\s+and\s+(?P<second>{_NUMERIC_TOKEN_PATTERN})",
    flags=re.IGNORECASE,
)
_NUMERIC_RANGE_RE = re.compile(
    rf"(?P<first>{_NUMERIC_TOKEN_PATTERN})\s*(?:-|to|and)\s*(?P<second>{_NUMERIC_TOKEN_PATTERN})",
    flags=re.IGNORECASE,
)
_NUMERIC_BELOW_RE = re.compile(
    rf"(?:<=|≤|<|below|under|at most|no more than)\s*(?P<value>{_NUMERIC_TOKEN_PATTERN})",
    flags=re.IGNORECASE,
)
_NUMERIC_ABOVE_RE = re.compile(
    rf"(?:>=|≥|>|above|over|at least|no less than)\s*(?P<value>{_NUMERIC_TOKEN_PATTERN})",
    flags=re.IGNORECASE,
)
_NUMERIC_SUFFIX_BELOW_RE = re.compile(
    rf"(?P<value>{_NUMERIC_TOKEN_PATTERN})\s*(?:or less|or lower|or below|or under)",
    flags=re.IGNORECASE,
)
_NUMERIC_SUFFIX_ABOVE_RE = re.compile(
    rf"(?P<value>{_NUMERIC_TOKEN_PATTERN})\s*(?:or more|or higher|or above|or over)",
    flags=re.IGNORECASE,
)
_NUMERIC_PLUS_RE = re.compile(
    rf"(?P<value>{_NUMERIC_TOKEN_PATTERN})\s*\+",
    flags=re.IGNORECASE,
)
_NUMERIC_SINGLE_RE = re.compile(
    rf"(?P<value>{_NUMERIC_TOKEN_PATTERN})",
    flags=re.IGNORECASE,
)
_NUMERIC_UNIT_SIGNAL_RE = re.compile(
    r"[$€£]|%|bps|bp|basis\s*points?|"
    r"\d\s*(?:k|m|b|t|bn|mm|mn|thousand|million|billion|trillion|percent|pct)",
    flags=re.IGNORECASE,
)
_NUMERIC_RANGE_SIGNAL_RE = re.compile(r"\d[\d,]*\s*-\s*\d")
_NUMERIC_BOUND_SIGNAL_RE = re.compile(
    r"(?:<=|≥|>=|≤|<|>|"
    r"\b(?:over|under|below|above|between|to|at least|at most|no more than|no less than|or more|or less)\b|"
    r"\+|↑|↓)",
    flags=re.IGNORECASE,
)


def _normalize_numeric_text(text: str) -> str:
    return text.translate(_DASH_TRANSLATION)


def _has_numeric_signal(text: str) -> bool:
    if not text:
        return False
    if not re.search(r"\d", text):
        return False
    normalized = _normalize_numeric_text(text)
    if _NUMERIC_BETWEEN_RE.search(normalized) or _NUMERIC_RANGE_RE.search(normalized):
        return True
    if (
        _NUMERIC_BELOW_RE.search(normalized)
        or _NUMERIC_ABOVE_RE.search(normalized)
        or _NUMERIC_SUFFIX_BELOW_RE.search(normalized)
        or _NUMERIC_SUFFIX_ABOVE_RE.search(normalized)
        or _NUMERIC_PLUS_RE.search(normalized)
    ):
        return True
    if _NUMERIC_UNIT_SIGNAL_RE.search(normalized):
        return True
    if _NUMERIC_RANGE_SIGNAL_RE.search(normalized):
        return True
    if _NUMERIC_BOUND_SIGNAL_RE.search(normalized):
        return True
    if _is_numeric_only(normalized):
        return True
    return False


def _is_numeric_only(text: str) -> bool:
    if not text or not re.search(r"\d", text):
        return False
    return not re.search(r"[a-zA-Z]", text)


def _numeric_range_similarity(left: str, right: str) -> Optional[float]:
    left_bounds = _parse_numeric_bounds(left)
    right_bounds = _parse_numeric_bounds(right)
    if not left_bounds or not right_bounds:
        return None
    if left_bounds.kind != right_bounds.kind:
        return None
    return _bounds_similarity(
        (left_bounds.low, left_bounds.high),
        (right_bounds.low, right_bounds.high),
    )


def _parse_numeric_bounds(text: str) -> Optional[_NumericBounds]:
    if not text:
        return None
    numeric_only = _is_numeric_only(text)
    if not _has_numeric_signal(text) and not numeric_only:
        return None
    normalized = _normalize_numeric_text(text)
    match = _NUMERIC_BETWEEN_RE.search(normalized)
    if match:
        return _parse_numeric_range(match.group("first"), match.group("second"))
    match = _NUMERIC_RANGE_RE.search(normalized)
    if match:
        return _parse_numeric_range(match.group("first"), match.group("second"))
    match = _NUMERIC_BELOW_RE.search(normalized) or _NUMERIC_SUFFIX_BELOW_RE.search(normalized)
    if match:
        token = _extract_numeric_token(match.group("value"))
        if not token:
            return None
        value, kind = _coerce_numeric_value(token, None)
        if value is None:
            return None
        return _NumericBounds(low=None, high=value, kind=kind)
    match = _NUMERIC_ABOVE_RE.search(normalized) or _NUMERIC_SUFFIX_ABOVE_RE.search(normalized)
    if match:
        token = _extract_numeric_token(match.group("value"))
        if not token:
            return None
        value, kind = _coerce_numeric_value(token, None)
        if value is None:
            return None
        return _NumericBounds(low=value, high=None, kind=kind)
    match = _NUMERIC_PLUS_RE.search(normalized)
    if match:
        token = _extract_numeric_token(match.group("value"))
        if not token:
            return None
        value, kind = _coerce_numeric_value(token, None)
        if value is None:
            return None
        return _NumericBounds(low=value, high=None, kind=kind)
    match = _NUMERIC_SINGLE_RE.search(normalized)
    if match:
        token = _extract_numeric_token(match.group("value"))
        if not token or (not token.has_unit and not numeric_only):
            return None
        value, kind = _coerce_numeric_value(token, None)
        if value is None:
            return None
        return _NumericBounds(low=value, high=value, kind=kind)
    return None


def _parse_numeric_range(first: str, second: str) -> Optional[_NumericBounds]:
    left = _extract_numeric_token(first)
    right = _extract_numeric_token(second)
    if not left or not right:
        return None
    if left.kind != "unknown" and right.kind != "unknown" and left.kind != right.kind:
        return None
    value_left, kind_left = _coerce_numeric_value(left, right if right.has_unit else None)
    value_right, kind_right = _coerce_numeric_value(right, left if left.has_unit else None)
    if value_left is None or value_right is None:
        return None
    if kind_left != kind_right:
        return None
    low, high = (value_left, value_right) if value_left <= value_right else (value_right, value_left)
    return _NumericBounds(low=low, high=high, kind=kind_left)


def _extract_numeric_token(text: str) -> Optional[_NumericToken]:
    match = _NUMERIC_VALUE_RE.search(text)
    if not match:
        return None
    raw_num = match.group("num")
    if not raw_num:
        return None
    try:
        value = float(raw_num.replace(",", ""))
    except (TypeError, ValueError):
        return None
    prefix = match.group("prefix") or ""
    unit_raw = (match.group("unit") or "").strip().lower()
    unit_clean = unit_raw.replace(" ", "")
    multiplier = 1.0
    kind = "unknown"
    has_unit = False
    if unit_clean in {"%", "percent", "pct"}:
        multiplier = 0.01
        kind = "percent"
        has_unit = True
    elif unit_clean in {"bps", "bp", "basispoints"}:
        multiplier = 0.0001
        kind = "percent"
        has_unit = True
    elif unit_clean in {"k", "thousand"}:
        multiplier = 1e3
        kind = "abs"
        has_unit = True
    elif unit_clean in {"m", "mm", "mn", "million"}:
        multiplier = 1e6
        kind = "abs"
        has_unit = True
    elif unit_clean in {"b", "bn", "billion"}:
        multiplier = 1e9
        kind = "abs"
        has_unit = True
    elif unit_clean in {"t", "tn", "trillion"}:
        multiplier = 1e12
        kind = "abs"
        has_unit = True
    elif prefix:
        multiplier = 1.0
        kind = "abs"
        has_unit = True
    return _NumericToken(
        raw_value=value,
        multiplier=multiplier,
        kind=kind,
        has_unit=has_unit,
    )


def _coerce_numeric_value(token: _NumericToken, fallback: Optional[_NumericToken]) -> tuple[Optional[float], str]:
    kind = token.kind
    multiplier = token.multiplier
    if token.kind == "unknown" and fallback is not None:
        kind = fallback.kind
        if fallback.has_unit:
            multiplier = fallback.multiplier
    if kind == "unknown":
        kind = "abs"
    return token.raw_value * multiplier, kind


def _to_number(raw: str) -> Optional[float]:
    try:
        return float(raw.replace(",", ""))
    except (TypeError, ValueError):
        return None


def _bounds_similarity(
    left: tuple[Optional[float], Optional[float]],
    right: tuple[Optional[float], Optional[float]],
) -> float:
    left_low, left_high = left
    right_low, right_high = right
    left_is_point = left_low is not None and left_high is not None and left_low == left_high
    right_is_point = right_low is not None and right_high is not None and right_low == right_high
    left_is_range = left_low is not None and left_high is not None and left_low < left_high
    right_is_range = right_low is not None and right_high is not None and right_low < right_high
    left_is_upper = left_low is None and left_high is not None
    right_is_upper = right_low is None and right_high is not None
    left_is_lower = left_low is not None and left_high is None
    right_is_lower = right_low is not None and right_high is None
    if left_is_point and right_is_point:
        return 1.0 if _point_within_tolerance(left_low, right_low) else 0.0
    if left_is_range and right_is_range:
        return _range_overlap_ratio(left_low, left_high, right_low, right_high)
    if left_low is None and left_high is not None and right_low is None and right_high is not None:
        return 1.0 if _point_within_tolerance(left_high, right_high) else 0.0
    if left_low is not None and left_high is None and right_low is not None and right_high is None:
        return 1.0 if _point_within_tolerance(left_low, right_low) else 0.0
    if left_is_range and (right_is_lower or right_is_upper):
        return _range_vs_bound(left_low, left_high, right_low, right_high)
    if right_is_range and (left_is_lower or left_is_upper):
        return _range_vs_bound(right_low, right_high, left_low, left_high)
    if left_is_point and right_is_range:
        if right_low <= left_low <= right_high:
            return 1.0
        nearest = right_low if left_low < right_low else right_high
        return _value_similarity(left_low, nearest)
    if right_is_point and left_is_range:
        if left_low <= right_low <= left_high:
            return 1.0
        nearest = left_low if right_low < left_low else left_high
        return _value_similarity(right_low, nearest)
    if left_is_point and (right_is_lower or right_is_upper):
        return _range_vs_bound(left_low, left_high, right_low, right_high)
    if right_is_point and (left_is_lower or left_is_upper):
        return _range_vs_bound(right_low, right_high, left_low, left_high)
    return 0.0


def _range_overlap_ratio(
    left_low: float,
    left_high: float,
    right_low: float,
    right_high: float,
) -> float:
    intersection = min(left_high, right_high) - max(left_low, right_low)
    if intersection <= 0:
        return 0.0
    union = max(left_high, right_high) - min(left_low, right_low)
    if union <= 0:
        return 1.0 if left_low == right_low and left_high == right_high else 0.0
    return intersection / union


def _range_vs_bound(
    range_low: float,
    range_high: float,
    bound_low: Optional[float],
    bound_high: Optional[float],
) -> float:
    span = max(range_high - range_low, 0.01)
    if bound_low is None and bound_high is not None:
        if range_high <= bound_high:
            return 1.0
        if range_low >= bound_high:
            return 0.0
        return max(0.0, (bound_high - range_low) / span)
    if bound_low is not None and bound_high is None:
        if range_low >= bound_low:
            return 1.0
        if range_high <= bound_low:
            return 0.0
        return max(0.0, (range_high - bound_low) / span)
    return 0.0


def _value_similarity(left: float, right: float) -> float:
    denom = max(max(left, right), 1e-6)
    return max(0.0, 1.0 - abs(left - right) / denom)


_POINT_VALUE_ABS_TOL = 0.01
_POINT_VALUE_REL_TOL = 0.01


def _point_within_tolerance(left: float, right: float) -> bool:
    diff = abs(left - right)
    denom = max(max(abs(left), abs(right)), 1e-6)
    limit = max(_POINT_VALUE_ABS_TOL, _POINT_VALUE_REL_TOL * denom)
    return diff <= limit


def _title_similarity(opinion_title: str, pm_title_norm: str) -> Optional[float]:
    if not opinion_title or not pm_title_norm:
        return None
    opinion_norm = normalize_text(opinion_title)
    if not opinion_norm:
        return None
    return SequenceMatcher(None, opinion_norm, pm_title_norm).ratio()


def _labels_similarity(
    opinion: OpinionMarketCandidate,
    pm_outcomes_norm: Optional[List[str]],
) -> Optional[float]:
    if not pm_outcomes_norm:
        return None
    labels = _opinion_label_candidates(opinion)
    if not labels:
        return None
    if len(labels) >= 2 and len(pm_outcomes_norm) >= 2:
        best = 0.0
        left = labels[0]
        right = labels[1]
        for i, outcome_left in enumerate(pm_outcomes_norm):
            for j, outcome_right in enumerate(pm_outcomes_norm):
                if i == j:
                    continue
                score = (_text_similarity(left, outcome_left) + _text_similarity(right, outcome_right)) / 2.0
                if score > best:
                    best = score
        return best
    label = labels[0]
    return max(_text_similarity(label, outcome) for outcome in pm_outcomes_norm)


def _text_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def _opinion_label_candidates(opinion: OpinionMarketCandidate) -> List[str]:
    yes_label = normalize_text(opinion.yes_label or "")
    no_label = normalize_text(opinion.no_label or "")
    if yes_label and no_label and not _is_generic_yes_no(yes_label, no_label):
        return [yes_label, no_label]
    child_title = normalize_text(opinion.title or "")
    if child_title:
        return [child_title]
    return []


def _is_generic_yes_no(yes_label: str, no_label: str) -> bool:
    return {yes_label, no_label} == {"yes", "no"}


def _teams_similarity(team_a: Optional[str], team_b: Optional[str], pm_teams_norm: Optional[List[str]]) -> Optional[float]:
    if not team_a or not team_b or not pm_teams_norm:
        return None
    opinion_teams = _normalize_teams(team_a, team_b)
    if not opinion_teams:
        return None
    set_op = set(opinion_teams)
    set_pm = set(pm_teams_norm)
    intersection = set_op.intersection(set_pm)
    if len(intersection) == 2:
        return 1.0
    if len(intersection) == 1:
        return 0.6
    return 0.0


def _time_similarity(
    op_start: Optional[datetime],
    op_end: Optional[datetime],
    pm_start: Optional[datetime],
    pm_end: Optional[datetime],
    config: MatchingConfig,
) -> Optional[float]:
    max_hours = config.max_time_diff_hours
    full_hours = max(config.time_full_score_hours, 0.01)
    best = None
    if op_start and pm_start:
        diff = abs((op_start - pm_start).total_seconds()) / 3600
        best = _time_score(diff, full_hours, max_hours)
    if op_end and pm_end:
        diff = abs((op_end - pm_end).total_seconds()) / 3600
        score = _time_score(diff, full_hours, max_hours)
        best = score if best is None else max(best, score)
    return best


def _time_score(diff_hours: float, full_hours: float, max_hours: float) -> float:
    if diff_hours <= full_hours:
        return 1.0
    if diff_hours >= max_hours:
        return 0.0
    span = max(max_hours - full_hours, 0.01)
    return max(0.0, 1.0 - ((diff_hours - full_hours) / span))


def _normalize_teams(team_a: Optional[str], team_b: Optional[str]) -> Optional[List[str]]:
    if not team_a or not team_b:
        return None
    left = normalize_text(team_a)
    right = normalize_text(team_b)
    if not left or not right:
        return None
    return [left, right]


def _normalize_outcomes(outcomes: List[str]) -> List[str]:
    cleaned: List[str] = []
    for outcome in outcomes:
        text = normalize_text(outcome or "")
        if text:
            cleaned.append(text)
    return cleaned


def _build_pair(
    opinion: OpinionMarketCandidate,
    pm_market: PolymarketMarket,
    confidence: float,
    match_details: dict,
    event_title_map: Optional[dict[str, str]] = None,
) -> MarketPair:
    pair_id = f"{opinion.market_id}:{pm_market.market_id}"
    status = "active"
    if pm_market.closed:
        status = "closed"
    elif not pm_market.accepting_orders:
        status = "paused"
    pm_yes_token = pm_market.clob_token_ids[0] if pm_market.clob_token_ids else ""
    pm_no_token = pm_market.clob_token_ids[1] if len(pm_market.clob_token_ids) > 1 else ""
    pm_event_title = None
    if event_title_map:
        pm_event_title = event_title_map.get(pm_market.market_id)
    if not pm_event_title:
        pm_event_title = pm_market.event_title or pm_market.question or None
    source_json = {
        "opinion": {
            "market_id": opinion.market_id,
            "event_volume": opinion.event_volume,
            "status": opinion.status,
            "market_type": opinion.market_type,
            "root_market_title": opinion.root_market_title,
        },
        "polymarket": {
            "market_id": pm_market.market_id,
            "event_title": pm_event_title,
            "question": pm_market.question,
            "group_item_title": pm_market.group_item_title,
            "slug": pm_market.slug,
            "condition_id": pm_market.condition_id,
            "closed": pm_market.closed,
            "accepting_orders": pm_market.accepting_orders,
        },
        "match": match_details,
    }
    return MarketPair(
        pair_id=pair_id,
        root_event_id=opinion.root_event_id,
        root_market_title=opinion.root_market_title,
        opinion_market_id=opinion.market_id,
        opinion_yes_token_id=opinion.yes_token_id,
        opinion_no_token_id=opinion.no_token_id,
        polymarket_market_id=pm_market.market_id,
        polymarket_event_title=pm_event_title,
        polymarket_condition_id=pm_market.condition_id,
        polymarket_yes_token_id=pm_yes_token,
        polymarket_no_token_id=pm_no_token,
        matching_confidence=round(confidence, 6),
        status=status,
        source_title=opinion.title,
        source_start_time=opinion.start_time,
        source_end_time=opinion.end_time,
        source_league=opinion.league,
        source_team_a=opinion.team_a,
        source_team_b=opinion.team_b,
        source_json=source_json,
    )


def _build_low_confidence_pair(
    opinion: OpinionMarketCandidate,
    pm_market: PolymarketMarket,
    confidence: float,
    match_details: dict,
    event_title_map: Optional[dict[str, str]] = None,
) -> LowConfidencePair:
    pair_id = f"{opinion.market_id}:{pm_market.market_id}"
    opinion_title = opinion.title or ""
    opinion_parent_title = opinion.root_market_title or opinion_title
    pm_event_title = None
    if event_title_map:
        pm_event_title = event_title_map.get(pm_market.market_id)
    if not pm_event_title:
        pm_event_title = pm_market.event_title or pm_market.question or None
    pm_title = pm_market.group_item_title or pm_market.question or pm_event_title or ""
    pm_parent_title = pm_event_title or pm_title
    source_json = {
        "opinion": {
            "market_id": opinion.market_id,
            "title": opinion.title,
            "root_market_title": opinion.root_market_title,
            "root_event_id": opinion.root_event_id,
            "event_volume": opinion.event_volume,
            "status": opinion.status,
            "market_type": opinion.market_type,
            "yes_label": opinion.yes_label,
            "no_label": opinion.no_label,
        },
        "polymarket": {
            "market_id": pm_market.market_id,
            "question": pm_market.question,
            "event_title": pm_event_title,
            "group_item_title": pm_market.group_item_title,
            "slug": pm_market.slug,
            "condition_id": pm_market.condition_id,
            "closed": pm_market.closed,
            "accepting_orders": pm_market.accepting_orders,
        },
        "match": match_details,
    }
    return LowConfidencePair(
        pair_id=pair_id,
        opinion_market_id=opinion.market_id,
        opinion_title=opinion_title,
        opinion_parent_title=opinion_parent_title,
        polymarket_market_id=pm_market.market_id,
        polymarket_title=pm_title,
        polymarket_parent_title=pm_parent_title,
        polymarket_slug=pm_market.slug,
        matching_confidence=round(confidence, 6),
        source_json=source_json,
    )


def _build_gate_filtered_pair(
    opinion: OpinionMarketCandidate,
    pm_market: PolymarketMarket,
    pm_meta: dict,
    gate_stage: str,
    gate_score: float,
    gate_threshold: float,
    event_title_score: Optional[float],
    child_title_score: Optional[float],
    gate_reasons: Optional[List[str]] = None,
    event_title_map: Optional[dict[str, str]] = None,
) -> GateFilteredPair:
    pair_id = f"{opinion.market_id}:{pm_market.market_id}"
    opinion_title = opinion.title or ""
    opinion_parent_title = opinion.root_market_title or opinion_title
    pm_event_title = None
    if event_title_map:
        pm_event_title = event_title_map.get(pm_market.market_id)
    if not pm_event_title:
        pm_event_title = pm_market.event_title or pm_market.question or None
    pm_title = pm_market.group_item_title or pm_market.question or pm_event_title or ""
    pm_parent_title = pm_event_title or pm_title
    numeric_score = _numeric_range_similarity(opinion.title or "", pm_market.question or "")
    source_json = {
        "opinion": {
            "market_id": opinion.market_id,
            "title": opinion.title,
            "root_market_title": opinion.root_market_title,
            "root_event_id": opinion.root_event_id,
            "event_volume": opinion.event_volume,
            "status": opinion.status,
            "market_type": opinion.market_type,
            "yes_label": opinion.yes_label,
            "no_label": opinion.no_label,
        },
        "polymarket": {
            "market_id": pm_market.market_id,
            "question": pm_market.question,
            "event_title": pm_event_title,
            "group_item_title": pm_market.group_item_title,
            "slug": pm_market.slug,
            "condition_id": pm_market.condition_id,
            "closed": pm_market.closed,
            "accepting_orders": pm_market.accepting_orders,
        },
        "gate": {
            "stage": gate_stage,
            "score": gate_score,
            "threshold": gate_threshold,
            "event_title": event_title_score,
            "child_title": child_title_score,
            "numeric_range": numeric_score,
            "reasons": gate_reasons or [],
        },
    }
    return GateFilteredPair(
        pair_id=pair_id,
        gate_stage=gate_stage,
        gate_score=round(gate_score, 6),
        gate_threshold=round(gate_threshold, 6),
        opinion_market_id=opinion.market_id,
        opinion_title=opinion_title,
        opinion_parent_title=opinion_parent_title,
        polymarket_market_id=pm_market.market_id,
        polymarket_title=pm_title,
        polymarket_parent_title=pm_parent_title,
        polymarket_slug=pm_market.slug,
        event_title_score=event_title_score,
        child_title_score=child_title_score,
        numeric_range_score=numeric_score,
        source_json=source_json,
    )


def _build_assignment_unmatched_pair(
    opinion: OpinionMarketCandidate,
    best_score: float,
    best_details: dict,
    min_confidence: float,
) -> GateFilteredPair:
    opinion_title = opinion.title or ""
    opinion_parent_title = opinion.root_market_title or opinion_title
    source_json = {
        "opinion": {
            "market_id": opinion.market_id,
            "title": opinion.title,
            "root_market_title": opinion.root_market_title,
            "root_event_id": opinion.root_event_id,
            "event_volume": opinion.event_volume,
            "status": opinion.status,
            "market_type": opinion.market_type,
            "yes_label": opinion.yes_label,
            "no_label": opinion.no_label,
        },
        "assignment": {
            "score": best_score,
            "min_confidence": min_confidence,
        },
    }
    return GateFilteredPair(
        pair_id=f"{opinion.market_id}:unmatched",
        gate_stage="assignment",
        gate_score=round(best_score, 6),
        gate_threshold=round(min_confidence, 6),
        opinion_market_id=opinion.market_id,
        opinion_title=opinion_title,
        opinion_parent_title=opinion_parent_title,
        polymarket_market_id="",
        polymarket_title="",
        polymarket_parent_title="",
        polymarket_slug=None,
        event_title_score=best_details.get("event_title"),
        child_title_score=best_details.get("title_child"),
        numeric_range_score=best_details.get("numeric_range"),
        source_json=source_json,
    )


def _score_value(details: Optional[dict], key: str) -> float:
    if not details:
        return 0.0
    value = details.get(key)
    return float(value) if value is not None else 0.0


def _polymarket_url(market: PolymarketMarket) -> str:
    if market.slug:
        return f"https://polymarket.com/market/{market.slug}"
    return f"https://gamma-api.polymarket.com/markets/{market.market_id}"


def _log_title(value: str, limit: int = 120) -> str:
    text = " ".join(value.split())
    if len(text) <= limit:
        return text
    return text[:limit]


def _opinion_event_title(opinion: OpinionMarketCandidate) -> str:
    root_title = (opinion.root_market_title or "").strip()
    if root_title:
        return root_title
    return (opinion.title or "").strip()


def _opinion_parent_key(opinion: OpinionMarketCandidate, parent_title: str) -> str:
    if opinion.root_event_id:
        return f"id:{opinion.root_event_id}"
    normalized = normalize_text(parent_title or "")
    if normalized:
        return f"title:{normalized}"
    return f"id:{opinion.market_id}"
