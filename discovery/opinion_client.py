from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import aiohttp

from .config import OpinionConfig
from .models import MarketPair, OpinionMarketCandidate
from .utils import parse_datetime, parse_yes_no_teams, safe_float, safe_int


class OpinionClient:
    def __init__(self, session: aiohttp.ClientSession, config: OpinionConfig, logger=None):
        self.session = session
        self.config = config
        self.logger = logger
        self._rate_limiter = _RateLimiter(self.config.max_rps)

    async def fetch_markets(self) -> Tuple[List[OpinionMarketCandidate], Dict[str, int]]:
        markets: List[OpinionMarketCandidate] = []
        stats: Dict[str, int] = {
            "markets_seen": 0,
            "markets_kept": 0,
            "status_filtered": 0,
            "not_binary": 0,
            "labels_ambiguous": 0,
            "title_parse_fail": 0,
            "market_id_missing": 0,
            "volume_filtered": 0,
            "event_closed": 0,
            "categorical_empty": 0,
            "categorical_fetch_failed": 0,
        }
        page = 1
        pages = 0
        total_items = 0
        raw_logged = False
        if self.logger:
            self.logger.info("opinion scan start")
        while page <= self.config.max_pages:
            payload = await self._fetch_page(page)
            items = payload.get("list") or payload.get("data") or []
            if self.logger and page == 1:
                _log_first_page_summary(self.logger, payload, items)
            if self.logger and not raw_logged and items:
                sample = items[0]
                sample_id = sample.get("marketId") or sample.get("market_id") or ""
                sample_json = _safe_json(sample, 2000)
                self.logger.info("opinion raw sample market_id=%s raw=%s", sample_id, sample_json)
                raw_logged = True
            pages += 1
            total_items += len(items)
            if self.logger:
                if page == 1 or len(items) < self.config.page_limit or page % 10 == 0:
                    self.logger.info("opinion page page=%d items=%d", page, len(items))
                else:
                    self.logger.debug("opinion page page=%d items=%d", page, len(items))
            if not items:
                break
            for item in items:
                candidates = await self._expand_candidates(item, stats)
                for candidate in candidates:
                    stats["markets_seen"] += 1
                    parsed = _parse_market(candidate, self.config)
                    if isinstance(parsed, OpinionMarketCandidate):
                        markets.append(parsed)
                        stats["markets_kept"] += 1
                    else:
                        stats[parsed] = stats.get(parsed, 0) + 1
            if len(items) < self.config.page_limit:
                break
            page += 1
            await asyncio.sleep(self.config.request_sleep_ms / 1000)
        if self.logger:
            self.logger.info("opinion scan done pages=%d items=%d kept=%d", pages, total_items, len(markets))
            if stats.get("categorical_empty") or stats.get("categorical_fetch_failed"):
                self.logger.info(
                    "opinion categorical skipped empty=%d fetch_failed=%d",
                    stats.get("categorical_empty", 0),
                    stats.get("categorical_fetch_failed", 0),
                )
            self.logger.info(
                "opinion reject summary market_id_missing=%d status_filtered=%d not_binary=%d labels_ambiguous=%d title_parse_fail=%d volume_filtered=%d event_closed=%d categorical_empty=%d categorical_fetch_failed=%d",
                stats.get("market_id_missing", 0),
                stats.get("status_filtered", 0),
                stats.get("not_binary", 0),
                stats.get("labels_ambiguous", 0),
                stats.get("title_parse_fail", 0),
                stats.get("volume_filtered", 0),
                stats.get("event_closed", 0),
                stats.get("categorical_empty", 0),
                stats.get("categorical_fetch_failed", 0),
            )
        return markets, stats

    async def _fetch_page(self, page: int) -> Dict[str, Any]:
        params = {
            "page": page,
            "limit": self.config.page_limit,
            "status": "activated",
            "marketType": self.config.market_type,
            "sortBy": self.config.sort_by,
        }
        headers = {"apikey": self.config.api_key}
        timeout = aiohttp.ClientTimeout(total=self.config.request_timeout_sec)
        payload = await self._request_json(self.config.base_url, params=params, headers=headers, timeout=timeout)
        result = payload.get("result") if isinstance(payload, dict) else None
        if isinstance(result, dict):
            return result
        return payload if isinstance(payload, dict) else {}

    async def enrich_pairs_with_orderbooks(self, pairs: List[MarketPair]) -> Dict[str, int]:
        stats = {"orderbooks_fetched": 0, "orderbook_errors": 0}
        for pair in pairs:
            yes_token = pair.opinion_yes_token_id
            no_token = pair.opinion_no_token_id
            if not yes_token or not no_token:
                continue
            try:
                yes_book = await self._fetch_orderbook(yes_token)
                no_book = await self._fetch_orderbook(no_token)
                summary = {
                    "yes": _summarize_orderbook(yes_book),
                    "no": _summarize_orderbook(no_book),
                }
                pair.source_json.setdefault("opinion_orderbook", {}).update(summary)
                stats["orderbooks_fetched"] += 2
            except Exception:
                stats["orderbook_errors"] += 1
        return stats

    async def _expand_candidates(
        self,
        raw: Dict[str, Any],
        stats: Dict[str, int],
    ) -> List[Dict[str, Any]]:
        children = raw.get("childMarkets") or raw.get("child_markets") or []
        if isinstance(children, list) and children:
            return _flatten_children(raw, children)
        if _is_categorical_market(raw):
            market_id = raw.get("marketId") or raw.get("market_id")
            if not market_id:
                stats["market_id_missing"] += 1
                return []
            try:
                detail = await self._fetch_categorical(str(market_id))
                child_list = detail.get("childMarkets") or detail.get("child_markets") or []
                if isinstance(child_list, list) and child_list:
                    return _flatten_children(raw, child_list)
                stats["categorical_empty"] += 1
                return []
            except Exception:
                stats["categorical_fetch_failed"] += 1
                return []
        return [raw]

    async def _fetch_categorical(self, market_id: str) -> Dict[str, Any]:
        url = f"{self.config.base_url.rstrip('/')}/categorical/{market_id}"
        headers = {"apikey": self.config.api_key}
        timeout = aiohttp.ClientTimeout(total=self.config.request_timeout_sec)
        payload = await self._request_json(url, headers=headers, timeout=timeout)
        result = payload.get("result") if isinstance(payload, dict) else None
        if isinstance(result, dict):
            return result
        return payload if isinstance(payload, dict) else {}

    async def _fetch_orderbook(self, token_id: str) -> Dict[str, Any]:
        url = _token_orderbook_url(self.config.base_url)
        headers = {"apikey": self.config.api_key}
        params = {"token_id": token_id}
        timeout = aiohttp.ClientTimeout(total=self.config.request_timeout_sec)
        payload = await self._request_json(url, params=params, headers=headers, timeout=timeout)
        result = payload.get("result") if isinstance(payload, dict) else None
        if isinstance(result, dict):
            return result
        return payload if isinstance(payload, dict) else {}

    async def _request_json(self, url: str, **kwargs) -> Dict[str, Any]:
        retries = 3
        backoff_base = 0.5
        for attempt in range(retries + 1):
            await self._rate_limiter.wait()
            try:
                async with self.session.get(url, **kwargs) as resp:
                    if resp.status >= 400:
                        text = await resp.text()
                        raise RuntimeError(f"opinion request failed ({resp.status}): {text}")
                    payload = await resp.json()
                return payload if isinstance(payload, dict) else {}
            except asyncio.TimeoutError:
                if attempt >= retries:
                    raise
                backoff = backoff_base * (2**attempt)
                if self.logger:
                    self.logger.warning(
                        "opinion request timeout retry=%d/%d backoff=%.1fs url=%s",
                        attempt + 1,
                        retries,
                        backoff,
                        url,
                    )
                await asyncio.sleep(backoff)
        return {}


def _parse_market(raw: Dict[str, Any], config: OpinionConfig) -> OpinionMarketCandidate | str:
    market_id = raw.get("marketId") or raw.get("market_id")
    if not market_id:
        return "market_id_missing"
    status_enum = str(raw.get("statusEnum") or "").strip()
    status_value = str(raw.get("status") or "").strip()
    status_num = safe_int(raw.get("status"))
    status_enum_lower = status_enum.lower()
    status_value_lower = status_value.lower()
    is_active = (
        status_num == 1
        or status_enum_lower in {"activated", "created"}
        or status_value_lower in {"activated", "created"}
    )
    if not is_active:
        return "status_filtered"
    status = status_enum or status_value
    market_type = safe_int(raw.get("marketType") if "marketType" in raw else raw.get("market_type"))
    yes_label = str(raw.get("yesLabel") or raw.get("yes_label") or "").strip()
    no_label = str(raw.get("noLabel") or raw.get("no_label") or "").strip()
    if not yes_label or not no_label:
        return "labels_ambiguous"
    if yes_label.lower() == no_label.lower():
        return "labels_ambiguous"
    if market_type is not None and market_type != 0 and not _is_yes_no(yes_label, no_label):
        return "not_binary"

    title = str(raw.get("marketTitle") or raw.get("title") or "").strip()
    team_a = None
    team_b = None
    if _is_yes_no(yes_label, no_label):
        teams = parse_yes_no_teams(title)
        if teams:
            team_a, team_b = teams[0], teams[1]
    else:
        team_a, team_b = yes_label, no_label

    event_volume = _extract_event_volume(raw)
    if event_volume is None or event_volume < config.min_event_volume:
        return "volume_filtered"
    end_time = _extract_end_time(raw)
    if end_time and end_time <= datetime.now(tz=timezone.utc):
        return "event_closed"

    return OpinionMarketCandidate(
        market_id=str(market_id),
        title=title,
        yes_label=yes_label,
        no_label=no_label,
        yes_token_id=str(raw.get("yesTokenId") or raw.get("yes_token_id") or ""),
        no_token_id=str(raw.get("noTokenId") or raw.get("no_token_id") or ""),
        event_volume=event_volume,
        status=status,
        market_type=market_type,
        root_event_id=_root_event_id(raw, market_id),
        root_market_id=_root_market_id(raw, market_id),
        root_market_title=_root_market_title(raw),
        start_time=_extract_start_time(raw),
        end_time=end_time,
        league=_extract_league(raw),
        team_a=team_a,
        team_b=team_b,
        raw=raw,
    )


def _log_first_page_summary(logger, payload: Dict[str, Any], items: List[Dict[str, Any]]) -> None:
    total = payload.get("total")
    market_type_counts: Dict[str, int] = {}
    child_total = 0
    child_parents = 0
    for item in items:
        market_type = safe_int(item.get("marketType") if "marketType" in item else item.get("market_type"))
        key = str(market_type) if market_type is not None else "none"
        market_type_counts[key] = market_type_counts.get(key, 0) + 1
        children = item.get("childMarkets") or item.get("child_markets") or []
        if isinstance(children, list) and children:
            child_parents += 1
            child_total += len(children)
    logger.info(
        "opinion page1 summary total=%s items=%d market_types=%s child_parents=%d child_markets=%d",
        total,
        len(items),
        _format_counter(market_type_counts),
        child_parents,
        child_total,
    )


def _extract_event_volume(raw: Dict[str, Any]) -> Optional[float]:
    for key in (
        "eventVolume",
        "eventVolumeUsd",
        "event_volume",
        "event_volume_usd",
        "volume",
        "volumeUsd",
        "volume_24h",
        "volume24h",
    ):
        value = safe_float(raw.get(key))
        if value is not None:
            return value
    return None


def _flatten_children(raw: Dict[str, Any], children: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    parent_id = raw.get("marketId") or raw.get("market_id")
    parent_title = str(raw.get("marketTitle") or raw.get("title") or "")
    parent_topic_id = raw.get("topicId") or raw.get("topic_id")
    parent_market_type = raw.get("marketType") if "marketType" in raw else raw.get("market_type")
    flattened: List[Dict[str, Any]] = []
    for child in children:
        if not isinstance(child, dict):
            continue
        child_copy = dict(child)
        child_copy["_parent_market_id"] = parent_id
        child_copy["_parent_market_title"] = parent_title
        if parent_topic_id is not None:
            child_copy["_parent_topic_id"] = parent_topic_id
        if parent_market_type is not None:
            child_copy["_parent_market_type"] = parent_market_type
        if not child_copy.get("marketTitle") and not child_copy.get("title"):
            if parent_title:
                child_copy["marketTitle"] = parent_title
        flattened.append(child_copy)
    return flattened


def _is_categorical_market(raw: Dict[str, Any]) -> bool:
    market_type = safe_int(raw.get("marketType") if "marketType" in raw else raw.get("market_type"))
    if market_type is not None and market_type != 0:
        return True
    is_categorical = raw.get("isCategorical") or raw.get("categorical")
    if isinstance(is_categorical, bool):
        return is_categorical
    return False


def _extract_start_time(raw: Dict[str, Any]) -> Optional[Any]:
    for key in (
        "startTime",
        "start_time",
        "eventStartTime",
        "event_start_time",
        "gameStartTime",
        "game_start_time",
        "beginAt",
        "begin_at",
        "startAt",
        "start_at",
    ):
        value = raw.get(key)
        if value is not None:
            return parse_datetime(value)
    return None


def _extract_end_time(raw: Dict[str, Any]) -> Optional[Any]:
    for key in (
        "endTime",
        "end_time",
        "endDate",
        "end_date",
        "settleTime",
        "settle_time",
        "expireTime",
        "expire_time",
        "cutoffAt",
        "cutoff_at",
        "closeTime",
        "close_time",
        "closeDate",
        "close_date",
    ):
        value = raw.get(key)
        if value is None:
            continue
        if isinstance(value, (int, float)) and value <= 0:
            continue
        if isinstance(value, str) and value.strip() in {"", "0"}:
            continue
        parsed = parse_datetime(value)
        if parsed:
            return parsed
    return None


def _extract_league(raw: Dict[str, Any]) -> str | None:
    for key in ("league", "tournament", "category", "sport"):
        value = raw.get(key)
        if isinstance(value, dict):
            name = value.get("name") or value.get("slug")
            if name:
                return str(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _root_market_id(raw: Dict[str, Any], market_id: str) -> str:
    return str(
        raw.get("rootMarketId")
        or raw.get("root_market_id")
        or raw.get("_parent_market_id")
        or market_id
    )


def _root_market_title(raw: Dict[str, Any]) -> str | None:
    title = raw.get("_parent_market_title") or raw.get("rootMarketTitle") or raw.get("root_market_title")
    if title:
        return str(title)
    fallback = raw.get("marketTitle") or raw.get("title")
    return str(fallback) if fallback else None


def _root_event_id(raw: Dict[str, Any], market_id: str) -> str:
    return str(
        raw.get("rootMarketId")
        or raw.get("root_market_id")
        or raw.get("_parent_market_id")
        or raw.get("eventId")
        or raw.get("event_id")
        or market_id
    )


def _is_yes_no(yes_label: str, no_label: str) -> bool:
    return {yes_label.strip().lower(), no_label.strip().lower()} == {"yes", "no"}


def _format_counter(counter: Dict[str, int]) -> str:
    if not counter:
        return "-"
    parts = [f"{key}:{counter[key]}" for key in sorted(counter.keys())]
    return ",".join(parts)


def _safe_json(value: Any, limit: int) -> str:
    try:
        text = json.dumps(value, ensure_ascii=True, separators=(",", ":"), default=str)
    except (TypeError, ValueError):
        text = str(value)
    return _truncate_text(text, limit)


def _truncate_text(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    return value[:limit]


def _token_orderbook_url(base_url: str) -> str:
    marker = "/openapi/market"
    if marker in base_url:
        root = base_url.split(marker)[0]
        return f"{root}/openapi/token/orderbook"
    return base_url.rstrip("/") + "/openapi/token/orderbook"


def _summarize_orderbook(payload: Dict[str, Any]) -> Dict[str, Any]:
    bids = _extract_levels(payload.get("bids"))
    asks = _extract_levels(payload.get("asks"))
    bids_sorted = sorted(bids, key=lambda item: item[0], reverse=True)
    asks_sorted = sorted(asks, key=lambda item: item[0])
    top_bids = bids_sorted[:3]
    top_asks = asks_sorted[:3]
    best_bid = top_bids[0][0] if top_bids else None
    best_ask = top_asks[0][0] if top_asks else None
    spread = (best_ask - best_bid) if best_ask is not None and best_bid is not None else None
    top3_depth = {
        "bids": round(sum(size for _, size in top_bids), 6) if top_bids else 0.0,
        "asks": round(sum(size for _, size in top_asks), 6) if top_asks else 0.0,
    }
    return {
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": spread,
        "top3_depth": top3_depth,
        "top_bids": [{"price": price, "size": size} for price, size in top_bids],
        "top_asks": [{"price": price, "size": size} for price, size in top_asks],
    }


def _extract_levels(raw_levels: Any) -> List[Tuple[float, float]]:
    levels: List[Tuple[float, float]] = []
    if not isinstance(raw_levels, list):
        return levels
    for item in raw_levels:
        price = None
        size = None
        if isinstance(item, dict):
            price = safe_float(item.get("price"))
            size = safe_float(item.get("amount") if "amount" in item else item.get("size"))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            price = safe_float(item[0])
            size = safe_float(item[1])
        if price is None or size is None:
            continue
        levels.append((price, size))
    return levels


class _RateLimiter:
    def __init__(self, max_rps: float):
        self.max_rps = max(1.0, float(max_rps))
        self.min_interval = 1.0 / self.max_rps
        self._lock = asyncio.Lock()
        self._last_call = 0.0

    async def wait(self) -> None:
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            delay = self.min_interval - elapsed
            if delay > 0:
                await asyncio.sleep(delay)
            self._last_call = time.monotonic()
