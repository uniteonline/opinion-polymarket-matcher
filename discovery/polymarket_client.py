from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Tuple

import aiohttp

from .config import PolymarketConfig
from .models import PolymarketEvent, PolymarketMarket
from .utils import parse_datetime, parse_json_list


class PolymarketClient:
    def __init__(self, session: aiohttp.ClientSession, config: PolymarketConfig, logger=None):
        self.session = session
        self.config = config
        self.logger = logger

    async def fetch_markets(self) -> Tuple[List[PolymarketMarket], Dict[str, int]]:
        markets: List[PolymarketMarket] = []
        stats: Dict[str, int] = {
            "markets_seen": 0,
            "markets_kept": 0,
            "closed_filtered": 0,
            "archived_filtered": 0,
            "inactive_filtered": 0,
            "missing_tokens": 0,
        }
        offset = 0
        pages = 0
        total_items = 0
        if self.logger:
            self.logger.info("polymarket scan start")
        while pages < self.config.max_pages:
            payload = await self._fetch_page(offset)
            items = payload if isinstance(payload, list) else []
            pages += 1
            total_items += len(items)
            if self.logger:
                if pages == 1 or len(items) < self.config.limit or pages % 10 == 0:
                    self.logger.info("polymarket page offset=%d items=%d", offset, len(items))
                else:
                    self.logger.debug("polymarket page offset=%d items=%d", offset, len(items))
            if not items:
                break
            for item in items:
                stats["markets_seen"] += 1
                parsed = _parse_market(item, self.config)
                if isinstance(parsed, PolymarketMarket):
                    markets.append(parsed)
                    stats["markets_kept"] += 1
                else:
                    stats[parsed] = stats.get(parsed, 0) + 1
            if len(items) < self.config.limit:
                break
            offset += self.config.limit
            await asyncio.sleep(self.config.request_sleep_ms / 1000)
        if self.logger:
            self.logger.info("polymarket scan done pages=%d items=%d kept=%d", pages, total_items, len(markets))
        return markets, stats

    async def search_events(self, query: str) -> Tuple[List[PolymarketEvent], Dict[str, int]]:
        stats: Dict[str, int] = {
            "events_seen": 0,
            "events_kept": 0,
            "events_filtered": 0,
            "markets_seen": 0,
            "markets_kept": 0,
            "closed_filtered": 0,
            "archived_filtered": 0,
            "inactive_filtered": 0,
            "missing_tokens": 0,
        }
        if self.logger:
            self.logger.info("polymarket public-search start query=%s", query)
        payload = await self._request_public_search(query=query, limit=self.config.search_limit)
        items = _extract_public_search_events(payload)
        events: List[PolymarketEvent] = []
        for event in items:
            stats["events_seen"] += 1
            if not _event_active(event, self.config):
                stats["events_filtered"] += 1
                continue
            parsed = _parse_event(event, self.config, stats)
            if parsed:
                events.append(parsed)
                stats["events_kept"] += 1
        if self.logger:
            self.logger.info(
                "polymarket public-search done query=%s events=%d kept=%d markets=%d",
                query,
                len(items),
                len(events),
                stats.get("markets_kept", 0),
            )
        return events, stats

    async def search_markets(self, query: str) -> Tuple[List[PolymarketMarket], Dict[str, int]]:
        markets: List[PolymarketMarket] = []
        stats: Dict[str, int] = {
            "markets_seen": 0,
            "markets_kept": 0,
            "closed_filtered": 0,
            "archived_filtered": 0,
            "inactive_filtered": 0,
            "missing_tokens": 0,
        }
        if self.logger:
            self.logger.info("polymarket search start query=%s", query)
        payload = await self._fetch_page(0, search=query, limit=self.config.search_limit)
        items = payload if isinstance(payload, list) else []
        for item in items:
            stats["markets_seen"] += 1
            parsed = _parse_market(item, self.config)
            if isinstance(parsed, PolymarketMarket):
                markets.append(parsed)
                stats["markets_kept"] += 1
            else:
                stats[parsed] = stats.get(parsed, 0) + 1
        if self.logger:
            self.logger.info(
                "polymarket search done query=%s items=%d kept=%d",
                query,
                len(items),
                len(markets),
            )
        return markets, stats

    async def _fetch_page(self, offset: int, search: str | None = None, limit: int | None = None) -> Any:
        params = {
            "limit": limit if limit is not None else self.config.limit,
            "offset": offset,
        }
        if search:
            params["search"] = search
        if not self.config.include_closed:
            params["closed"] = "false"
        if not self.config.include_archived:
            params["archived"] = "false"
        if self.config.active_only:
            params["active"] = "true"
        return await self._request_markets(params)

    async def _request_markets(self, params: Dict[str, Any]) -> Any:
        timeout = aiohttp.ClientTimeout(total=self.config.request_timeout_sec)
        url = f"{self.config.base_url.rstrip('/')}/markets"
        async with self.session.get(url, params=params, timeout=timeout) as resp:
            if resp.status >= 400:
                text = await resp.text()
                raise RuntimeError(f"polymarket request failed ({resp.status}): {text}")
            return await resp.json()

    async def _request_public_search(self, query: str, limit: int) -> Any:
        params = {"q": query}
        if limit > 0:
            params["limit"] = limit
        params.update(_normalize_public_search_params(self.config.public_search_params))
        timeout = aiohttp.ClientTimeout(total=self.config.request_timeout_sec)
        url = f"{self.config.base_url.rstrip('/')}/public-search"
        max_retries = max(0, int(self.config.public_search_retry_max))
        backoff_base = max(0.1, float(self.config.public_search_retry_base_sec))
        for attempt in range(max_retries + 1):
            async with self.session.get(
                url,
                params=params,
                timeout=timeout,
                headers={"User-Agent": "Mozilla/5.0"},
            ) as resp:
                if resp.status == 403:
                    text = await resp.text()
                    if attempt < max_retries:
                        delay = backoff_base * (2**attempt)
                        if self.logger:
                            self.logger.warning(
                                "polymarket public-search blocked status=%d retry=%d/%d delay=%.1fs query=%s",
                                resp.status,
                                attempt + 1,
                                max_retries,
                                delay,
                                query,
                            )
                        await asyncio.sleep(delay)
                        continue
                    raise RuntimeError(f"polymarket public-search request failed ({resp.status}): {text}")
                if resp.status >= 400:
                    text = await resp.text()
                    raise RuntimeError(f"polymarket public-search request failed ({resp.status}): {text}")
                return await resp.json()


def _event_active(raw: Dict[str, Any], config: PolymarketConfig) -> bool:
    closed = bool(raw.get("closed") or raw.get("isClosed"))
    archived = bool(raw.get("archived") or raw.get("isArchived"))
    active = raw.get("active")
    if active is None:
        active = raw.get("isActive")
    if not config.include_closed and closed:
        return False
    if not config.include_archived and archived:
        return False
    if config.active_only and active is not None and not bool(active):
        return False
    return True


def _parse_event(raw: Dict[str, Any], config: PolymarketConfig, stats: Dict[str, int]) -> PolymarketEvent | None:
    event_id = raw.get("id") or raw.get("event_id")
    if not event_id:
        return None
    title = str(raw.get("title") or raw.get("name") or raw.get("question") or "").strip()
    markets_raw = raw.get("markets") or []
    markets: List[PolymarketMarket] = []
    if isinstance(markets_raw, list):
        for entry in markets_raw:
            if not isinstance(entry, dict):
                continue
            stats["markets_seen"] += 1
            parsed = _parse_market(entry, config)
            if isinstance(parsed, PolymarketMarket):
                markets.append(parsed)
                stats["markets_kept"] += 1
            else:
                stats[parsed] = stats.get(parsed, 0) + 1
    return PolymarketEvent(
        event_id=str(event_id),
        title=title,
        markets=markets,
        raw=raw,
    )


def _parse_market(raw: Dict[str, Any], config: PolymarketConfig) -> PolymarketMarket | str:
    market_id = raw.get("id") or raw.get("market_id") or raw.get("marketId")
    if not market_id:
        return "missing_tokens"
    closed = bool(raw.get("closed") or raw.get("isClosed"))
    archived = bool(raw.get("archived") or raw.get("isArchived"))
    active = raw.get("active")
    if active is None:
        active = raw.get("isActive")
    if not config.include_closed and closed:
        return "closed_filtered"
    if not config.include_archived and archived:
        return "archived_filtered"
    if config.active_only and active is not None and not bool(active):
        return "inactive_filtered"

    clob_token_ids = parse_json_list(raw.get("clobTokenIds") or raw.get("clob_token_ids"))
    outcomes = parse_json_list(raw.get("outcomes") or raw.get("shortOutcomes"))
    if len(clob_token_ids) < 2 or len(outcomes) < 2:
        return "missing_tokens"
    question = str(raw.get("question") or raw.get("title") or raw.get("marketTitle") or "").strip()
    group_item_title = raw.get("groupItemTitle") or raw.get("group_item_title") or ""
    group_item_title = str(group_item_title).strip() or None
    event_title = raw.get("eventTitle") or raw.get("event_title")
    if event_title:
        event_title = str(event_title).strip() or None
    if not event_title:
        events = raw.get("events")
        if isinstance(events, list) and events:
            first_event = events[0]
            if isinstance(first_event, dict):
                raw_title = first_event.get("title") or first_event.get("name") or first_event.get("question")
                if raw_title:
                    event_title = str(raw_title).strip() or None
    slug = str(raw.get("slug") or raw.get("marketSlug") or raw.get("market_slug") or "").strip() or None
    condition_id = str(raw.get("conditionId") or raw.get("condition_id") or "")
    start_time = _extract_start_time(raw)
    end_time = _extract_end_time(raw)
    league = _extract_league(raw)
    team_a, team_b = _extract_teams(outcomes, question)
    accepting_orders = bool(
        raw.get("acceptingOrders")
        or raw.get("accepting_orders")
        or raw.get("acceptingOrdersTimestamp")
    )
    return PolymarketMarket(
        market_id=str(market_id),
        question=question,
        group_item_title=group_item_title,
        event_title=event_title,
        slug=slug,
        condition_id=condition_id,
        clob_token_ids=clob_token_ids,
        outcomes=outcomes,
        start_time=start_time,
        end_time=end_time,
        league=league,
        team_a=team_a,
        team_b=team_b,
        closed=closed,
        accepting_orders=accepting_orders,
        raw=raw,
    )


def _extract_start_time(raw: Dict[str, Any]):
    for key in ("startDate", "start_date", "startTime", "start_time", "eventStartTime", "event_start_time"):
        value = raw.get(key)
        if value is not None:
            return parse_datetime(value)
    return None


def _extract_end_time(raw: Dict[str, Any]):
    for key in ("endDate", "end_date", "closeDate", "close_date", "endTime", "end_time", "closeTime", "close_time"):
        value = raw.get(key)
        if value is not None:
            return parse_datetime(value)
    return None


def _extract_league(raw: Dict[str, Any]) -> str | None:
    for key in ("league", "sport", "category", "tournament"):
        value = raw.get(key)
        if isinstance(value, dict):
            name = value.get("name") or value.get("slug")
            if name:
                return str(name)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _extract_teams(outcomes: List[str], question: str) -> Tuple[str | None, str | None]:
    if len(outcomes) >= 2:
        return outcomes[0], outcomes[1]
    if question and " vs " in question.lower():
        parts = question.split(" vs ")
        if len(parts) >= 2:
            return parts[0].strip(), parts[1].strip()
    return None, None


def _extract_public_search_events(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    for key in ("events", "data", "results"):
        value = payload.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
        if isinstance(value, dict):
            events = value.get("events")
            if isinstance(events, list):
                return [item for item in events if isinstance(item, dict)]
    return []


def _normalize_public_search_params(params: Any) -> Dict[str, Any]:
    if not isinstance(params, dict):
        return {}
    normalized: Dict[str, Any] = {}
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, bool):
            normalized[key] = "true" if value else "false"
        else:
            normalized[key] = value
    return normalized
