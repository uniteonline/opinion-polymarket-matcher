from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional


@dataclass(slots=True)
class OpinionMarketCandidate:
    market_id: str
    title: str
    yes_label: str
    no_label: str
    yes_token_id: str
    no_token_id: str
    event_volume: float
    status: str
    market_type: Optional[int]
    root_event_id: Optional[str]
    root_market_id: Optional[str]
    root_market_title: Optional[str]
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    league: Optional[str]
    team_a: Optional[str]
    team_b: Optional[str]
    raw: Dict


@dataclass(slots=True)
class PolymarketMarket:
    market_id: str
    question: str
    group_item_title: Optional[str]
    event_title: Optional[str]
    slug: Optional[str]
    condition_id: str
    clob_token_ids: List[str]
    outcomes: List[str]
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    league: Optional[str]
    team_a: Optional[str]
    team_b: Optional[str]
    closed: bool
    accepting_orders: bool
    raw: Dict


@dataclass(slots=True)
class PolymarketEvent:
    event_id: str
    title: str
    markets: List[PolymarketMarket]
    raw: Dict


@dataclass(slots=True)
class MarketPair:
    pair_id: str
    root_event_id: Optional[str]
    root_market_title: Optional[str]
    opinion_market_id: str
    opinion_yes_token_id: str
    opinion_no_token_id: str
    polymarket_market_id: str
    polymarket_event_title: Optional[str]
    polymarket_condition_id: str
    polymarket_yes_token_id: str
    polymarket_no_token_id: str
    matching_confidence: float
    status: str
    source_title: str
    source_start_time: Optional[datetime]
    source_end_time: Optional[datetime]
    source_league: Optional[str]
    source_team_a: Optional[str]
    source_team_b: Optional[str]
    source_json: Dict


@dataclass(slots=True)
class LowConfidencePair:
    pair_id: str
    opinion_market_id: str
    opinion_title: str
    opinion_parent_title: str
    polymarket_market_id: str
    polymarket_title: str
    polymarket_parent_title: str
    polymarket_slug: Optional[str]
    matching_confidence: float
    source_json: Dict


@dataclass(slots=True)
class GateFilteredPair:
    pair_id: str
    gate_stage: str
    gate_score: float
    gate_threshold: float
    opinion_market_id: str
    opinion_title: str
    opinion_parent_title: str
    polymarket_market_id: str
    polymarket_title: str
    polymarket_parent_title: str
    polymarket_slug: Optional[str]
    event_title_score: Optional[float]
    child_title_score: Optional[float]
    numeric_range_score: Optional[float]
    source_json: Dict
