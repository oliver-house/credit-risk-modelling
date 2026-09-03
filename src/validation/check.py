import argparse
import sys

import pandas as pd

from src.config import DATA_FILES
from src.utils.helpers import get_logger, timer
from src.validation.schemas import TABLE_SCHEMAS, validate_frames

logger = get_logger(__name__)


def load_frames(keys: list[str], data_dir=None, n_rows: int = 0) -> dict:
    frames = {}
    for key in keys:
        path = DATA_FILES[key]
        if data_dir is not None:
            path = data_dir / TABLE_SCHEMAS[key].filename
        if not path.exists():
            logger.error(f"{key}: {path} not found")
            continue
        with timer(f"Reading {key}", logger):
            frames[key] = pd.read_csv(path, nrows=n_rows or None)
        logger.info(f"  {key}: {frames[key].shape}")
    return frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tables", nargs="*", default=sorted(TABLE_SCHEMAS),
                        choices=sorted(TABLE_SCHEMAS),
                        help="Tables to validate (default: all eight)")
    parser.add_argument("--data-dir", default=None,
                        help="Directory holding the CSVs (default: the configured one)")
    parser.add_argument("--rows", type=int, default=0,
                        help="Read only the first N rows of each table")
    args = parser.parse_args()

    from pathlib import Path
    data_dir = Path(args.data_dir) if args.data_dir else None

    frames = load_frames(args.tables, data_dir, args.rows)
    if not frames:
        logger.error("No tables could be read")
        return 2

    violations = validate_frames(frames)
    errors = [v for v in violations if v.severity == "error"]
    warnings = [v for v in violations if v.severity == "warn"]

    for v in warnings:
        logger.warning(str(v))
    for v in errors:
        logger.error(str(v))

    logger.info(f"{len(frames)} table(s) checked: "
                f"{len(errors)} error(s), {len(warnings)} warning(s)")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
