from __future__ import annotations

import csv
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass(slots=True)
class LowConfidenceMatch:
    watchlist_version: int
    opinion_market_id: str
    opinion_title: str
    opinion_parent_title: str
    polymarket_market_id: str
    polymarket_title: str
    polymarket_parent_title: str
    polymarket_url: str
    matching_confidence: float

    def as_dict(self) -> dict:
        return {
            "watchlist_version": self.watchlist_version,
            "opinion_market_id": self.opinion_market_id,
            "opinion_title": self.opinion_title,
            "opinion_parent_title": self.opinion_parent_title,
            "polymarket_market_id": self.polymarket_market_id,
            "polymarket_title": self.polymarket_title,
            "polymarket_parent_title": self.polymarket_parent_title,
            "polymarket_url": self.polymarket_url,
            "matching_confidence": self.matching_confidence,
        }


@dataclass(slots=True)
class GateFilteredMatch:
    watchlist_version: int
    gate_stage: str
    gate_score: float
    gate_threshold: float
    opinion_market_id: str
    opinion_title: str
    opinion_parent_title: str
    polymarket_market_id: str
    polymarket_title: str
    polymarket_parent_title: str
    polymarket_url: str
    event_title_score: Optional[float]
    child_title_score: Optional[float]
    numeric_range_score: Optional[float]

    def as_dict(self) -> dict:
        return {
            "watchlist_version": self.watchlist_version,
            "gate_stage": self.gate_stage,
            "gate_score": self.gate_score,
            "gate_threshold": self.gate_threshold,
            "opinion_market_id": self.opinion_market_id,
            "opinion_title": self.opinion_title,
            "opinion_parent_title": self.opinion_parent_title,
            "polymarket_market_id": self.polymarket_market_id,
            "polymarket_title": self.polymarket_title,
            "polymarket_parent_title": self.polymarket_parent_title,
            "polymarket_url": self.polymarket_url,
            "event_title_score": self.event_title_score,
            "child_title_score": self.child_title_score,
            "numeric_range_score": self.numeric_range_score,
        }


class LowConfidenceReviewHelper:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path).expanduser()

    @classmethod
    def from_config(cls, config_path: str = "discovery/config.yaml") -> "LowConfidenceReviewHelper":
        from .config import load_config

        config = load_config(config_path)
        return cls(config.database_path)

    def get_latest_version(self) -> int:
        return self._resolve_version(None)

    def list_below_threshold(
        self,
        threshold: Optional[float] = None,
        version: Optional[int] = None,
    ) -> List[LowConfidenceMatch]:
        target_version = self._resolve_version(version)
        rows = self._fetch_rows(target_version, threshold)
        results: List[LowConfidenceMatch] = []
        for row in rows:
            results.append(self._row_to_match(row))
        return results

    def write_csv(
        self,
        output_path: str,
        threshold: Optional[float] = None,
        version: Optional[int] = None,
    ) -> int:
        matches = self.list_below_threshold(threshold, version=version)
        output = Path(output_path).expanduser()
        fieldnames = [
            "watchlist_version",
            "opinion_market_id",
            "opinion_title",
            "opinion_parent_title",
            "polymarket_market_id",
            "polymarket_title",
            "polymarket_parent_title",
            "polymarket_url",
            "matching_confidence",
        ]
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for match in matches:
                writer.writerow(match.as_dict())
        return len(matches)

    def list_gate_filtered(
        self,
        version: Optional[int] = None,
        stage: Optional[str] = None,
    ) -> List[GateFilteredMatch]:
        target_version = self._resolve_version(version)
        rows = self._fetch_gate_filtered_rows(target_version, stage)
        results: List[GateFilteredMatch] = []
        for row in rows:
            results.append(self._row_to_gate_match(row))
        return results

    def write_gate_filtered_csv(
        self,
        output_path: str,
        version: Optional[int] = None,
        stage: Optional[str] = None,
    ) -> int:
        matches = self.list_gate_filtered(version=version, stage=stage)
        output = Path(output_path).expanduser()
        fieldnames = [
            "watchlist_version",
            "gate_stage",
            "gate_score",
            "gate_threshold",
            "opinion_market_id",
            "opinion_title",
            "opinion_parent_title",
            "polymarket_market_id",
            "polymarket_title",
            "polymarket_parent_title",
            "polymarket_url",
            "event_title_score",
            "child_title_score",
            "numeric_range_score",
        ]
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for match in matches:
                writer.writerow(match.as_dict())
        return len(matches)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _resolve_version(self, version: Optional[int]) -> int:
        if version is not None:
            return int(version)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT value FROM watchlist_meta WHERE key = ?",
                ("watchlist_version",),
            ).fetchone()
            if row and row["value"]:
                try:
                    return int(row["value"])
                except (TypeError, ValueError):
                    pass
            row = conn.execute("SELECT MAX(version) AS version FROM watchlist_versions").fetchone()
            if row and row["version"] is not None:
                return int(row["version"])
        return 0

    def _fetch_rows(self, version: int, threshold: Optional[float]) -> Iterable[sqlite3.Row]:
        sql = """
            SELECT
                watchlist_version,
                opinion_market_id,
                opinion_title,
                opinion_parent_title,
                polymarket_market_id,
                polymarket_title,
                polymarket_parent_title,
                polymarket_slug,
                matching_confidence,
                source_json
            FROM low_confidence_pairs
            WHERE watchlist_version = ?
        """
        params = [version]
        if threshold is not None:
            sql += " AND matching_confidence < ?"
            params.append(float(threshold))
        sql += " ORDER BY matching_confidence ASC"
        with self._connect() as conn:
            table = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='low_confidence_pairs'"
            ).fetchone()
            if not table:
                raise RuntimeError("low_confidence_pairs table missing; run discovery once to populate it")
            return list(conn.execute(sql, params).fetchall())

    def _fetch_gate_filtered_rows(
        self,
        version: int,
        stage: Optional[str],
    ) -> Iterable[sqlite3.Row]:
        sql = """
            SELECT
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
                source_json
            FROM gate_filtered_pairs
            WHERE watchlist_version = ?
        """
        params = [version]
        if stage:
            sql += " AND gate_stage = ?"
            params.append(stage)
        sql += " ORDER BY gate_score DESC"
        with self._connect() as conn:
            table = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='gate_filtered_pairs'"
            ).fetchone()
            if not table:
                raise RuntimeError("gate_filtered_pairs table missing; run discovery once to populate it")
            return list(conn.execute(sql, params).fetchall())

    def _row_to_match(self, row: sqlite3.Row) -> LowConfidenceMatch:
        source_json = self._safe_json(row["source_json"])
        opinion_info = source_json.get("opinion", {}) if isinstance(source_json, dict) else {}
        polymarket_info = source_json.get("polymarket", {}) if isinstance(source_json, dict) else {}

        opinion_title = self._clean_title(row["opinion_title"]) or self._clean_title(opinion_info.get("title"))
        opinion_parent = self._clean_title(row["opinion_parent_title"]) or self._clean_title(
            opinion_info.get("root_market_title")
        )

        pm_title = self._clean_title(row["polymarket_title"]) or self._clean_title(
            polymarket_info.get("group_item_title") or polymarket_info.get("question")
        )
        pm_parent = self._clean_title(row["polymarket_parent_title"]) or self._clean_title(
            polymarket_info.get("event_title")
        )
        if not pm_parent:
            pm_parent = pm_title

        pm_slug = self._clean_title(row["polymarket_slug"]) or self._clean_title(polymarket_info.get("slug"))
        pm_url = self._polymarket_url(pm_slug, row["polymarket_market_id"])

        return LowConfidenceMatch(
            watchlist_version=int(row["watchlist_version"]),
            opinion_market_id=str(row["opinion_market_id"]),
            opinion_title=opinion_title,
            opinion_parent_title=opinion_parent,
            polymarket_market_id=str(row["polymarket_market_id"]),
            polymarket_title=pm_title,
            polymarket_parent_title=pm_parent,
            polymarket_url=pm_url,
            matching_confidence=float(row["matching_confidence"]),
        )

    def _row_to_gate_match(self, row: sqlite3.Row) -> GateFilteredMatch:
        source_json = self._safe_json(row["source_json"])
        opinion_info = source_json.get("opinion", {}) if isinstance(source_json, dict) else {}
        polymarket_info = source_json.get("polymarket", {}) if isinstance(source_json, dict) else {}

        opinion_title = self._clean_title(row["opinion_title"]) or self._clean_title(opinion_info.get("title"))
        opinion_parent = self._clean_title(row["opinion_parent_title"]) or self._clean_title(
            opinion_info.get("root_market_title")
        )

        pm_title = self._clean_title(row["polymarket_title"]) or self._clean_title(
            polymarket_info.get("group_item_title") or polymarket_info.get("question")
        )
        pm_parent = self._clean_title(row["polymarket_parent_title"]) or self._clean_title(
            polymarket_info.get("event_title")
        )
        if not pm_parent:
            pm_parent = pm_title

        pm_slug = self._clean_title(row["polymarket_slug"]) or self._clean_title(polymarket_info.get("slug"))
        pm_url = self._polymarket_url(pm_slug, row["polymarket_market_id"])

        return GateFilteredMatch(
            watchlist_version=int(row["watchlist_version"]),
            gate_stage=str(row["gate_stage"]),
            gate_score=float(row["gate_score"]),
            gate_threshold=float(row["gate_threshold"]),
            opinion_market_id=str(row["opinion_market_id"]),
            opinion_title=opinion_title,
            opinion_parent_title=opinion_parent,
            polymarket_market_id=str(row["polymarket_market_id"]),
            polymarket_title=pm_title,
            polymarket_parent_title=pm_parent,
            polymarket_url=pm_url,
            event_title_score=_safe_float(row["event_title_score"]),
            child_title_score=_safe_float(row["child_title_score"]),
            numeric_range_score=_safe_float(row["numeric_range_score"]),
        )

    @staticmethod
    def _polymarket_url(slug: str, market_id: str) -> str:
        if slug:
            return f"https://polymarket.com/market/{slug}"
        return f"https://gamma-api.polymarket.com/markets/{market_id}"

    @staticmethod
    def _safe_json(raw: str) -> dict:
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _clean_title(value: Optional[str]) -> str:
        if not value:
            return ""
        return " ".join(str(value).split())


def _safe_float(value: Optional[object]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
