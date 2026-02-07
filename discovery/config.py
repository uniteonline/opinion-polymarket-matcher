from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass(slots=True)
class OpinionConfig:
    base_url: str = "https://openapi.opinion.trade/openapi/market"
    api_key: str = ""
    request_timeout_sec: int = 30
    page_limit: int = 20
    max_pages: int = 50
    request_sleep_ms: int = 100
    min_event_volume: float = 20_000_000.0
    market_type: int = 2
    sort_by: int = 2
    max_rps: float = 15.0


@dataclass(slots=True)
class PolymarketConfig:
    base_url: str = "https://gamma-api.polymarket.com"
    request_timeout_sec: int = 30
    limit: int = 200
    max_pages: int = 50
    request_sleep_ms: int = 80
    include_closed: bool = False
    include_archived: bool = False
    active_only: bool = True
    search_mode: str = "hybrid"
    search_limit: int = 200
    public_search_retry_max: int = 4
    public_search_retry_base_sec: float = 3.0
    public_search_params: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MatchingConfig:
    min_confidence: float = 0.6
    title_gate_min: float = 0.45
    child_title_gate_min: float = 0.45
    weight_title: float = 0.5
    weight_labels: float = 0.3
    weight_teams: float = 0.3
    weight_time: float = 0.2
    max_time_diff_hours: float = 12.0
    time_full_score_hours: float = 1.0


@dataclass(slots=True)
class ScheduleConfig:
    interval_minutes: int = 60


@dataclass(slots=True)
class LoggingConfig:
    level: str = "INFO"


@dataclass(slots=True)
class DiscoveryConfig:
    database_path: str
    opinion: OpinionConfig
    polymarket: PolymarketConfig
    matching: MatchingConfig
    schedule: ScheduleConfig
    logging: LoggingConfig


def load_config(path: str) -> DiscoveryConfig:
    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        base = Path(__file__).resolve().parents[1]
        config_path = (base / config_path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}

    opinion_raw = raw.get("opinion", {})
    opinion_api_key = str(opinion_raw.get("api_key", "")).strip()
    if not opinion_api_key:
        opinion_api_key = os.getenv("OPINION_OPENAPI_KEY", "").strip()
    opinion = OpinionConfig(
        base_url=str(opinion_raw.get("base_url", OpinionConfig.base_url)),
        api_key=opinion_api_key,
        request_timeout_sec=int(opinion_raw.get("request_timeout_sec", OpinionConfig.request_timeout_sec)),
        page_limit=int(opinion_raw.get("page_limit", OpinionConfig.page_limit)),
        max_pages=int(opinion_raw.get("max_pages", OpinionConfig.max_pages)),
        request_sleep_ms=int(opinion_raw.get("request_sleep_ms", OpinionConfig.request_sleep_ms)),
        min_event_volume=float(opinion_raw.get("min_event_volume", OpinionConfig.min_event_volume)),
        market_type=int(opinion_raw.get("market_type", OpinionConfig.market_type)),
        sort_by=int(opinion_raw.get("sort_by", OpinionConfig.sort_by)),
        max_rps=float(opinion_raw.get("max_rps", OpinionConfig.max_rps)),
    )

    polymarket_raw = raw.get("polymarket", {})
    public_search_params = polymarket_raw.get("public_search_params", {})
    if not isinstance(public_search_params, dict):
        public_search_params = {}
    polymarket = PolymarketConfig(
        base_url=str(polymarket_raw.get("base_url", PolymarketConfig.base_url)),
        request_timeout_sec=int(
            polymarket_raw.get("request_timeout_sec", PolymarketConfig.request_timeout_sec)
        ),
        limit=int(polymarket_raw.get("limit", PolymarketConfig.limit)),
        max_pages=int(polymarket_raw.get("max_pages", PolymarketConfig.max_pages)),
        request_sleep_ms=int(polymarket_raw.get("request_sleep_ms", PolymarketConfig.request_sleep_ms)),
        include_closed=bool(polymarket_raw.get("include_closed", PolymarketConfig.include_closed)),
        include_archived=bool(polymarket_raw.get("include_archived", PolymarketConfig.include_archived)),
        active_only=bool(polymarket_raw.get("active_only", PolymarketConfig.active_only)),
        search_mode=str(polymarket_raw.get("search_mode", PolymarketConfig.search_mode)),
        search_limit=int(polymarket_raw.get("search_limit", PolymarketConfig.search_limit)),
        public_search_retry_max=int(
            polymarket_raw.get("public_search_retry_max", PolymarketConfig.public_search_retry_max)
        ),
        public_search_retry_base_sec=float(
            polymarket_raw.get("public_search_retry_base_sec", PolymarketConfig.public_search_retry_base_sec)
        ),
        public_search_params=public_search_params,
    )

    matching_raw = raw.get("matching", {})
    matching = MatchingConfig(
        min_confidence=float(matching_raw.get("min_confidence", MatchingConfig.min_confidence)),
        title_gate_min=float(matching_raw.get("title_gate_min", MatchingConfig.title_gate_min)),
        child_title_gate_min=float(
            matching_raw.get("child_title_gate_min", MatchingConfig.child_title_gate_min)
        ),
        weight_title=float(matching_raw.get("weight_title", MatchingConfig.weight_title)),
        weight_labels=float(matching_raw.get("weight_labels", MatchingConfig.weight_labels)),
        weight_teams=float(matching_raw.get("weight_teams", MatchingConfig.weight_teams)),
        weight_time=float(matching_raw.get("weight_time", MatchingConfig.weight_time)),
        max_time_diff_hours=float(
            matching_raw.get("max_time_diff_hours", MatchingConfig.max_time_diff_hours)
        ),
        time_full_score_hours=float(
            matching_raw.get("time_full_score_hours", MatchingConfig.time_full_score_hours)
        ),
    )

    schedule_raw = raw.get("schedule", {})
    schedule = ScheduleConfig(
        interval_minutes=int(schedule_raw.get("interval_minutes", ScheduleConfig.interval_minutes))
    )

    logging_raw = raw.get("logging", {})
    logging = LoggingConfig(level=str(logging_raw.get("level", LoggingConfig.level)))

    database_path = _resolve_path(raw, "database_path")
    return DiscoveryConfig(
        database_path=database_path,
        opinion=opinion,
        polymarket=polymarket,
        matching=matching,
        schedule=schedule,
        logging=logging,
    )


def _resolve_path(raw: Dict[str, Any], key: str) -> str:
    value = raw.get(key, "")
    if not value:
        raise ValueError(f"{key} missing in discovery config")
    path = Path(str(value)).expanduser()
    if path.is_absolute():
        return str(path)
    base = Path(__file__).resolve().parents[1]
    return str((base / path).resolve())
