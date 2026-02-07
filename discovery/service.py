from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Optional

import aiohttp

from .config import DiscoveryConfig
from .db import DiscoveryDatabase
from .matcher import match_markets
from .opinion_client import OpinionClient
from .polymarket_client import PolymarketClient
from .utils import normalize_text


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s:%(name)s:%(message)s",
    )


class DiscoveryService:
    def __init__(self, config: DiscoveryConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger("discovery")
        self.db = DiscoveryDatabase(config.database_path)

    async def run_once(self) -> None:
        await self.db.init()
        try:
            now = datetime.now(tz=timezone.utc)
            run_id = now.isoformat()
            self.logger.info("discovery run start run_id=%s", run_id)
            async with aiohttp.ClientSession() as session:
                opinion_client = OpinionClient(session, self.config.opinion, logger=self.logger)
                polymarket_client = PolymarketClient(session, self.config.polymarket, logger=self.logger)
                opinion_markets, opinion_stats = await opinion_client.fetch_markets()
                search_mode = (self.config.polymarket.search_mode or "scan").lower()
                if search_mode not in {"scan", "search", "hybrid"}:
                    self.logger.warning("unknown search_mode=%s, defaulting to scan", search_mode)
                    search_mode = "scan"
                polymarket_markets, polymarket_stats = [], {
                    "markets_seen": 0,
                    "markets_kept": 0,
                    "closed_filtered": 0,
                    "archived_filtered": 0,
                    "inactive_filtered": 0,
                    "missing_tokens": 0,
                }
                if search_mode in {"scan", "hybrid"}:
                    polymarket_markets, polymarket_stats = await polymarket_client.fetch_markets()
                search_cache: dict[str, list] = {}
                candidate_map: dict[str, list] | None = None
                event_title_map: dict[str, str] = {}
                if search_mode in {"search", "hybrid"}:
                    candidate_map = {}
                    search_markets: list = []
                    for opinion in opinion_markets:
                        root_id = opinion.root_event_id or opinion.market_id
                        if root_id in search_cache:
                            continue
                        query = (opinion.root_market_title or opinion.title or "").strip()
                        if not query:
                            search_cache[root_id] = []
                            continue
                        events, _ = await polymarket_client.search_events(query)
                        search_cache[root_id] = events
                        search_markets.extend(events)
                        if self.config.polymarket.request_sleep_ms:
                            await asyncio.sleep(self.config.polymarket.request_sleep_ms / 1000)
                    if search_markets:
                        known_ids = {market.market_id for market in polymarket_markets}
                        for event in search_markets:
                            for market in event.markets:
                                if market.market_id not in known_ids:
                                    polymarket_markets.append(market)
                                    known_ids.add(market.market_id)
                                if market.market_id not in event_title_map and event.title:
                                    event_title_map[market.market_id] = event.title
                    for root_id, events in search_cache.items():
                        if not events:
                            if search_mode == "search":
                                candidate_map[root_id] = []
                            continue
                        query = ""
                        for opinion in opinion_markets:
                            if (opinion.root_event_id or opinion.market_id) == root_id:
                                query = (opinion.root_market_title or opinion.title or "").strip()
                                break
                        passed_markets = []
                        for event in events:
                            if _title_similarity(query, event.title) < self.config.matching.title_gate_min:
                                continue
                            passed_markets.extend(event.markets)
                        if passed_markets:
                            deduped = {}
                            for market in passed_markets:
                                deduped[market.market_id] = market
                            candidate_map[root_id] = list(deduped.values())
                        else:
                            candidate_map[root_id] = []
                pairs, match_stats, low_confidence, gate_filtered = match_markets(
                    opinion_markets=opinion_markets,
                    polymarket_markets=polymarket_markets,
                    config=self.config.matching,
                    logger=self.logger,
                    candidate_map=candidate_map,
                    event_title_map=event_title_map,
                )
                orderbook_stats = await opinion_client.enrich_pairs_with_orderbooks(pairs)
                current_version = await self.db.get_current_version()
                next_version = current_version + 1
                notes = json.dumps(
                    {
                        "min_event_volume": self.config.opinion.min_event_volume,
                        "min_confidence": self.config.matching.min_confidence,
                        "orderbooks_fetched": orderbook_stats.get("orderbooks_fetched", 0),
                        "orderbook_errors": orderbook_stats.get("orderbook_errors", 0),
                    },
                    separators=(",", ":"),
                )
                opinion_filtered = int(opinion_stats.get("volume_filtered", 0))
                await self.db.store_watchlist(
                    version=next_version,
                    pairs=pairs,
                    opinion_markets=int(opinion_stats.get("markets_seen", 0)),
                    opinion_filtered=opinion_filtered,
                    polymarket_markets=int(polymarket_stats.get("markets_seen", 0)),
                    matches=match_stats.matched,
                    notes=notes,
                )
                await self.db.store_low_confidence_pairs(
                    version=next_version,
                    pairs=low_confidence,
                )
                await self.db.store_gate_filtered_pairs(
                    version=next_version,
                    pairs=gate_filtered,
                )
                self.logger.info(
                    "discovery run complete version=%d opinion_seen=%d opinion_filtered=%d polymarket_seen=%d pairs=%d",
                    next_version,
                    int(opinion_stats.get("markets_seen", 0)),
                    opinion_filtered,
                    int(polymarket_stats.get("markets_seen", 0)),
                    match_stats.matched,
                )
                if orderbook_stats.get("orderbook_errors"):
                    self.logger.warning(
                        "opinion orderbook errors=%d",
                        orderbook_stats.get("orderbook_errors", 0),
                    )
                self.logger.info("watchlist_updated version=%d", next_version)
        finally:
            await self.db.close()


def _title_similarity(left: str, right: str) -> float:
    left_norm = normalize_text(left or "")
    right_norm = normalize_text(right or "")
    if not left_norm or not right_norm:
        return 0.0
    return SequenceMatcher(None, left_norm, right_norm).ratio()


async def run_once(config: DiscoveryConfig) -> None:
    service = DiscoveryService(config)
    await service.run_once()


async def run_forever(config: DiscoveryConfig, interval_minutes: int) -> None:
    service = DiscoveryService(config)
    while True:
        try:
            await service.run_once()
        except asyncio.CancelledError:
            raise
        except Exception:
            service.logger.exception("discovery run failed")
        await asyncio.sleep(interval_minutes * 60)
