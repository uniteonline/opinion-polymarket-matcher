from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import aiosqlite

from .models import GateFilteredPair, LowConfidencePair, MarketPair
from .utils import to_timestamp


class DiscoveryDatabase:
    def __init__(self, path: str):
        self.path = Path(path)
        self._conn: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(self.path)
        self._conn.row_factory = aiosqlite.Row

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

    async def init(self) -> None:
        if not self._conn:
            await self.connect()
        assert self._conn is not None
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA synchronous=NORMAL")
        await self._conn.execute("PRAGMA busy_timeout=5000")
        await self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS watchlist_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        await self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS watchlist_versions (
                version INTEGER PRIMARY KEY,
                created_at_utc INTEGER NOT NULL,
                opinion_markets INTEGER NOT NULL,
                opinion_filtered INTEGER NOT NULL,
                polymarket_markets INTEGER NOT NULL,
                matches INTEGER NOT NULL,
                notes TEXT
            )
            """
        )
        await self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS market_pairs (
                pair_id TEXT PRIMARY KEY,
                watchlist_version INTEGER NOT NULL,
                root_event_id TEXT,
                root_market_title TEXT,
                opinion_market_id TEXT,
                opinion_yes_token_id TEXT,
                opinion_no_token_id TEXT,
                polymarket_market_id TEXT,
                polymarket_event_title TEXT,
                polymarket_condition_id TEXT,
                polymarket_yes_token_id TEXT,
                polymarket_no_token_id TEXT,
                matching_confidence REAL,
                status TEXT,
                created_at_utc INTEGER NOT NULL,
                updated_at_utc INTEGER NOT NULL,
                source_title TEXT,
                source_start_time_utc INTEGER,
                source_end_time_utc INTEGER,
                source_league TEXT,
                source_team_a TEXT,
                source_team_b TEXT,
                source_json TEXT
            )
            """
        )
        await self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS low_confidence_pairs (
                pair_id TEXT PRIMARY KEY,
                watchlist_version INTEGER NOT NULL,
                opinion_market_id TEXT NOT NULL,
                opinion_title TEXT,
                opinion_parent_title TEXT,
                polymarket_market_id TEXT NOT NULL,
                polymarket_title TEXT,
                polymarket_parent_title TEXT,
                polymarket_slug TEXT,
                matching_confidence REAL NOT NULL,
                created_at_utc INTEGER NOT NULL,
                updated_at_utc INTEGER NOT NULL,
                source_json TEXT
            )
            """
        )
        await self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS gate_filtered_pairs (
                pair_id TEXT PRIMARY KEY,
                watchlist_version INTEGER NOT NULL,
                gate_stage TEXT NOT NULL,
                gate_score REAL NOT NULL,
                gate_threshold REAL NOT NULL,
                opinion_market_id TEXT NOT NULL,
                opinion_title TEXT,
                opinion_parent_title TEXT,
                polymarket_market_id TEXT NOT NULL,
                polymarket_title TEXT,
                polymarket_parent_title TEXT,
                polymarket_slug TEXT,
                event_title_score REAL,
                child_title_score REAL,
                numeric_range_score REAL,
                created_at_utc INTEGER NOT NULL,
                updated_at_utc INTEGER NOT NULL,
                source_json TEXT
            )
            """
        )
        await self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_market_pairs_version ON market_pairs (watchlist_version)"
        )
        await self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_market_pairs_opinion ON market_pairs (opinion_market_id)"
        )
        await self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_market_pairs_polymarket ON market_pairs (polymarket_market_id)"
        )
        await self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_low_conf_pairs_version ON low_confidence_pairs (watchlist_version)"
        )
        await self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_gate_filtered_pairs_version ON gate_filtered_pairs (watchlist_version)"
        )
        await self._ensure_column("market_pairs", "root_market_title", "TEXT")
        await self._ensure_column("market_pairs", "polymarket_event_title", "TEXT")
        await self._conn.commit()

    async def get_current_version(self) -> int:
        if not self._conn:
            await self.connect()
        assert self._conn is not None
        cursor = await self._conn.execute(
            "SELECT value FROM watchlist_meta WHERE key = ?",
            ("watchlist_version",),
        )
        row = await cursor.fetchone()
        if not row:
            return 0
        try:
            return int(row["value"])
        except (TypeError, ValueError):
            return 0

    async def set_current_version(self, version: int) -> None:
        if not self._conn:
            await self.connect()
        assert self._conn is not None
        await self._conn.execute(
            """
            INSERT INTO watchlist_meta (key, value)
            VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = excluded.value
            """,
            ("watchlist_version", str(version)),
        )

    async def record_version(
        self,
        version: int,
        opinion_markets: int,
        opinion_filtered: int,
        polymarket_markets: int,
        matches: int,
        notes: Optional[str] = None,
    ) -> None:
        if not self._conn:
            await self.connect()
        assert self._conn is not None
        now_ts = int(datetime.now(tz=timezone.utc).timestamp())
        await self._conn.execute(
            """
            INSERT INTO watchlist_versions (
                version,
                created_at_utc,
                opinion_markets,
                opinion_filtered,
                polymarket_markets,
                matches,
                notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                version,
                now_ts,
                opinion_markets,
                opinion_filtered,
                polymarket_markets,
                matches,
                notes,
            ),
        )

    async def store_watchlist(
        self,
        version: int,
        pairs: Iterable[MarketPair],
        opinion_markets: int,
        opinion_filtered: int,
        polymarket_markets: int,
        matches: int,
        notes: Optional[str] = None,
    ) -> None:
        if not self._conn:
            await self.connect()
        assert self._conn is not None
        now_ts = int(datetime.now(tz=timezone.utc).timestamp())
        rows = []
        for pair in pairs:
            rows.append(
                (
                    pair.pair_id,
                    version,
                    pair.root_event_id,
                    pair.root_market_title,
                    pair.opinion_market_id,
                    pair.opinion_yes_token_id,
                    pair.opinion_no_token_id,
                    pair.polymarket_market_id,
                    pair.polymarket_event_title,
                    pair.polymarket_condition_id,
                    pair.polymarket_yes_token_id,
                    pair.polymarket_no_token_id,
                    pair.matching_confidence,
                    pair.status,
                    now_ts,
                    now_ts,
                    pair.source_title,
                    to_timestamp(pair.source_start_time),
                    to_timestamp(pair.source_end_time),
                    pair.source_league,
                    pair.source_team_a,
                    pair.source_team_b,
                    json.dumps(pair.source_json, separators=(",", ":"), ensure_ascii=False),
                )
            )
        await self._conn.execute("BEGIN")
        try:
            if rows:
                await self._conn.executemany(
                    """
                    INSERT INTO market_pairs (
                        pair_id,
                        watchlist_version,
                        root_event_id,
                        root_market_title,
                        opinion_market_id,
                        opinion_yes_token_id,
                        opinion_no_token_id,
                        polymarket_market_id,
                        polymarket_event_title,
                        polymarket_condition_id,
                        polymarket_yes_token_id,
                        polymarket_no_token_id,
                        matching_confidence,
                        status,
                        created_at_utc,
                        updated_at_utc,
                        source_title,
                        source_start_time_utc,
                        source_end_time_utc,
                        source_league,
                        source_team_a,
                        source_team_b,
                        source_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(pair_id) DO UPDATE SET
                        watchlist_version = excluded.watchlist_version,
                        root_event_id = excluded.root_event_id,
                        root_market_title = excluded.root_market_title,
                        opinion_market_id = excluded.opinion_market_id,
                        opinion_yes_token_id = excluded.opinion_yes_token_id,
                        opinion_no_token_id = excluded.opinion_no_token_id,
                        polymarket_market_id = excluded.polymarket_market_id,
                        polymarket_event_title = excluded.polymarket_event_title,
                        polymarket_condition_id = excluded.polymarket_condition_id,
                        polymarket_yes_token_id = excluded.polymarket_yes_token_id,
                        polymarket_no_token_id = excluded.polymarket_no_token_id,
                        matching_confidence = excluded.matching_confidence,
                        status = excluded.status,
                        updated_at_utc = excluded.updated_at_utc,
                        source_title = excluded.source_title,
                        source_start_time_utc = excluded.source_start_time_utc,
                        source_end_time_utc = excluded.source_end_time_utc,
                        source_league = excluded.source_league,
                        source_team_a = excluded.source_team_a,
                        source_team_b = excluded.source_team_b,
                        source_json = excluded.source_json
                    """,
                    rows,
                )
            await self._conn.execute(
                """
                INSERT INTO watchlist_versions (
                    version,
                    created_at_utc,
                    opinion_markets,
                    opinion_filtered,
                    polymarket_markets,
                    matches,
                    notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    version,
                    now_ts,
                    opinion_markets,
                    opinion_filtered,
                    polymarket_markets,
                    matches,
                    notes,
                ),
            )
            await self._conn.execute(
                """
                INSERT INTO watchlist_meta (key, value)
                VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                ("watchlist_version", str(version)),
            )
            await self._conn.commit()
        except Exception:
            await self._conn.rollback()
            raise

    async def store_low_confidence_pairs(
        self,
        version: int,
        pairs: Iterable[LowConfidencePair],
    ) -> None:
        if not self._conn:
            await self.connect()
        assert self._conn is not None
        now_ts = int(datetime.now(tz=timezone.utc).timestamp())
        rows = []
        for pair in pairs:
            rows.append(
                (
                    pair.pair_id,
                    version,
                    pair.opinion_market_id,
                    pair.opinion_title,
                    pair.opinion_parent_title,
                    pair.polymarket_market_id,
                    pair.polymarket_title,
                    pair.polymarket_parent_title,
                    pair.polymarket_slug,
                    pair.matching_confidence,
                    now_ts,
                    now_ts,
                    json.dumps(pair.source_json, separators=(",", ":"), ensure_ascii=False),
                )
            )
        if not rows:
            return
        await self._conn.execute("BEGIN")
        try:
            await self._conn.executemany(
                """
                INSERT INTO low_confidence_pairs (
                    pair_id,
                    watchlist_version,
                    opinion_market_id,
                    opinion_title,
                    opinion_parent_title,
                    polymarket_market_id,
                    polymarket_title,
                    polymarket_parent_title,
                    polymarket_slug,
                    matching_confidence,
                    created_at_utc,
                    updated_at_utc,
                    source_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(pair_id) DO UPDATE SET
                    watchlist_version = excluded.watchlist_version,
                    opinion_market_id = excluded.opinion_market_id,
                    opinion_title = excluded.opinion_title,
                    opinion_parent_title = excluded.opinion_parent_title,
                    polymarket_market_id = excluded.polymarket_market_id,
                    polymarket_title = excluded.polymarket_title,
                    polymarket_parent_title = excluded.polymarket_parent_title,
                    polymarket_slug = excluded.polymarket_slug,
                    matching_confidence = excluded.matching_confidence,
                    updated_at_utc = excluded.updated_at_utc,
                    source_json = excluded.source_json
                """,
                rows,
            )
            await self._conn.commit()
        except Exception:
            await self._conn.rollback()
            raise

    async def store_gate_filtered_pairs(
        self,
        version: int,
        pairs: Iterable[GateFilteredPair],
    ) -> None:
        if not self._conn:
            await self.connect()
        assert self._conn is not None
        now_ts = int(datetime.now(tz=timezone.utc).timestamp())
        rows = []
        for pair in pairs:
            rows.append(
                (
                    pair.pair_id,
                    version,
                    pair.gate_stage,
                    pair.gate_score,
                    pair.gate_threshold,
                    pair.opinion_market_id,
                    pair.opinion_title,
                    pair.opinion_parent_title,
                    pair.polymarket_market_id,
                    pair.polymarket_title,
                    pair.polymarket_parent_title,
                    pair.polymarket_slug,
                    pair.event_title_score,
                    pair.child_title_score,
                    pair.numeric_range_score,
                    now_ts,
                    now_ts,
                    json.dumps(pair.source_json, separators=(",", ":"), ensure_ascii=False),
                )
            )
        if not rows:
            return
        await self._conn.execute("BEGIN")
        try:
            await self._conn.executemany(
                """
                INSERT INTO gate_filtered_pairs (
                    pair_id,
                    watchlist_version,
                    gate_stage,
                    gate_score,
                    gate_threshold,
                    opinion_market_id,
                    opinion_title,
                    opinion_parent_title,
                    polymarket_market_id,
                    polymarket_title,
                    polymarket_parent_title,
                    polymarket_slug,
                    event_title_score,
                    child_title_score,
                    numeric_range_score,
                    created_at_utc,
                    updated_at_utc,
                    source_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(pair_id) DO UPDATE SET
                    watchlist_version = excluded.watchlist_version,
                    gate_stage = excluded.gate_stage,
                    gate_score = excluded.gate_score,
                    gate_threshold = excluded.gate_threshold,
                    opinion_market_id = excluded.opinion_market_id,
                    opinion_title = excluded.opinion_title,
                    opinion_parent_title = excluded.opinion_parent_title,
                    polymarket_market_id = excluded.polymarket_market_id,
                    polymarket_title = excluded.polymarket_title,
                    polymarket_parent_title = excluded.polymarket_parent_title,
                    polymarket_slug = excluded.polymarket_slug,
                    event_title_score = excluded.event_title_score,
                    child_title_score = excluded.child_title_score,
                    numeric_range_score = excluded.numeric_range_score,
                    updated_at_utc = excluded.updated_at_utc,
                    source_json = excluded.source_json
                """,
                rows,
            )
            await self._conn.commit()
        except Exception:
            await self._conn.rollback()
            raise

    async def _ensure_column(self, table: str, column: str, column_type: str) -> None:
        if not self._conn:
            await self.connect()
        assert self._conn is not None
        cursor = await self._conn.execute(f"PRAGMA table_info({table})")
        rows = await cursor.fetchall()
        existing = {row[1] for row in rows}
        if column in existing:
            return
        await self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")
