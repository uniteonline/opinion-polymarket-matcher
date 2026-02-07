from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Iterable, List, Optional, Tuple


_normalize_re = re.compile(r"[^\w]+", flags=re.UNICODE)


def normalize_text(value: str) -> str:
    text = value.strip().lower()
    text = text.replace("&", " ")
    text = text.replace("_", " ")
    text = _normalize_re.sub(" ", text)
    return " ".join(text.split())


def parse_datetime(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc)
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OverflowError, ValueError):
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(timezone.utc)
        except ValueError:
            return None
    return None


def to_timestamp(value: Optional[datetime]) -> Optional[int]:
    if not value:
        return None
    return int(value.replace(tzinfo=timezone.utc).timestamp())


def parse_json_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if item is not None]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            if "," in text:
                return [item.strip() for item in text.split(",") if item.strip()]
            return [text]
        if isinstance(parsed, list):
            return [str(item) for item in parsed if item is not None]
        return [str(parsed)]
    return [str(value)]


def join_text(parts: Iterable[Optional[str]]) -> str:
    return " ".join([part.strip() for part in parts if part and part.strip()])


def parse_yes_no_teams(text: str) -> Optional[Tuple[str, str]]:
    question = text.strip()
    if not question:
        return None
    patterns = [
        re.compile(
            r"^will\s+(?P<a>.+?)\s+(?:beat|defeat)\s+(?P<b>.+?)\??$",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"^will\s+(?P<a>.+?)\s+win\s+(?:vs|versus|against)\s+(?P<b>.+?)\??$",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"^(?P<a>.+?)\s+to\s+win\s+(?:vs|versus|against)\s+(?P<b>.+?)\??$",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"^(?P<a>.+?)\s+vs\.?\s+(?P<b>.+?)$",
            flags=re.IGNORECASE,
        ),
    ]
    for pattern in patterns:
        match = pattern.match(question)
        if match:
            left = match.group("a").strip()
            right = match.group("b").strip()
            if left and right:
                return left, right
    return None


def safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
