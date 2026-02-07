from __future__ import annotations

import argparse
from pathlib import Path

from .review_helper import LowConfidenceReviewHelper


DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "records"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export low-confidence discovery matches to CSV.")
    parser.add_argument(
        "--config",
        default="discovery/config.yaml",
        help="Path to discovery config yaml.",
    )
    parser.add_argument("--version", type=int, help="Watchlist version to export (default: latest)")
    parser.add_argument("--threshold", type=float, help="Optional upper bound on matching_confidence")
    parser.add_argument(
        "--mode",
        choices=("auto", "low_confidence", "gate_filtered"),
        default="auto",
        help="Export mode (default: auto).",
    )
    parser.add_argument(
        "--stage",
        choices=("parent", "child"),
        help="Filter gate_filtered rows by stage.",
    )
    parser.add_argument("--output", help="Override output CSV path")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    helper = LowConfidenceReviewHelper.from_config(args.config)
    version = args.version if args.version is not None else helper.get_latest_version()
    mode = args.mode
    if mode == "auto":
        low_rows = helper.list_below_threshold(version=version)
        mode = "low_confidence" if low_rows else "gate_filtered"
    if args.output:
        output_path = Path(args.output).expanduser()
    else:
        prefix = "low_confidence" if mode == "low_confidence" else "gate_filtered"
        output_path = DEFAULT_OUTPUT_DIR / f"{prefix}_v{version}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "gate_filtered":
        count = helper.write_gate_filtered_csv(str(output_path), version=version, stage=args.stage)
    else:
        count = helper.write_csv(str(output_path), threshold=args.threshold, version=version)
    print(f"wrote {count} rows to {output_path}")


if __name__ == "__main__":
    main()
