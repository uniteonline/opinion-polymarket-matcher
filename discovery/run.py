from __future__ import annotations

import argparse
import asyncio

from .config import load_config
from .service import configure_logging, run_forever, run_once


def main() -> None:
    parser = argparse.ArgumentParser(description="Discovery service (Opinion + Polymarket).")
    parser.add_argument(
        "--config",
        default="discovery/config.yaml",
        help="Path to discovery config yaml.",
    )
    parser.add_argument("--interval-minutes", type=int, help="Override run interval minutes")
    parser.add_argument("--once", action="store_true", help="Run one cycle and exit")
    args = parser.parse_args()

    config = load_config(args.config)
    configure_logging(config.logging.level)

    if args.once:
        asyncio.run(run_once(config))
        return
    interval = args.interval_minutes or config.schedule.interval_minutes
    asyncio.run(run_forever(config, interval))


if __name__ == "__main__":
    main()
