#!/usr/bin/env python3
"""Plot all signals from P73 real robot CSV logs.

Usage examples:
  python3 realrobot_26xxxx.py
  python3 realrobot_26xxxx.py --csv realrobot_260402_110559.csv
  python3 realrobot_26xxxx.py --csv ./realrobot_260402_110559.csv --show
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Avoid matplotlib cache permission issues in restricted environments.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt  # noqa: E402


INDEXED_COL_RE = re.compile(r"^(?P<prefix>.+)_(?P<index>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot all data from realrobot_26*.csv")
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="CSV file path. If omitted, latest realrobot_26*.csv in this script directory is used.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for PNG files. Default: plots_<csv_stem>",
    )
    parser.add_argument("--show", action="store_true", help="Open plot windows as well")
    parser.add_argument(
        "--max-subplots",
        type=int,
        default=12,
        help="Maximum subplot count per figure for indexed signals.",
    )
    return parser.parse_args()


def find_latest_csv(script_dir: Path) -> Path:
    candidates = sorted(script_dir.glob("realrobot_26*.csv"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"No file matching realrobot_26*.csv found in {script_dir}")
    return candidates[-1]


def load_csv_numeric(csv_path: Path) -> Tuple[List[str], Dict[str, np.ndarray]]:
    with csv_path.open("r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        ncols = len(header)

        rows: List[List[float]] = []
        for row in reader:
            if not row:
                continue
            # Some logs can end with a partially written row.
            if len(row) < ncols:
                row = row + ["nan"] * (ncols - len(row))
            elif len(row) > ncols:
                row = row[:ncols]
            rows.append([safe_float(v) for v in row])

    arr = np.array(rows, dtype=float)
    data = {name: arr[:, i] for i, name in enumerate(header)}

    if "time" in data and len(data["time"]) > 0:
        data["time"] = data["time"] - data["time"][0]

    return header, data


def safe_float(value: str) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        return float("nan")


def split_columns(header: List[str]) -> Tuple[List[str], Dict[str, List[str]]]:
    scalar_cols: List[str] = []
    indexed: Dict[str, List[Tuple[int, str]]] = {}

    for col in header:
        if col == "time":
            continue

        match = INDEXED_COL_RE.match(col)
        if match:
            prefix = match.group("prefix")
            idx = int(match.group("index"))
            indexed.setdefault(prefix, []).append((idx, col))
        else:
            scalar_cols.append(col)

    indexed_cols: Dict[str, List[str]] = {}
    for prefix, pairs in indexed.items():
        pairs.sort(key=lambda x: x[0])
        indexed_cols[prefix] = [name for _, name in pairs]

    return scalar_cols, indexed_cols


def plot_scalar_group(time: np.ndarray, data: Dict[str, np.ndarray], cols: List[str], out_path: Path) -> None:
    if not cols:
        return

    n = len(cols)
    ncols = 3
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, max(3.5 * nrows, 4)), sharex=True)
    axes = np.array(axes).reshape(-1)
    fig.suptitle("Scalar Signals", fontsize=14)

    for i, col in enumerate(cols):
        ax = axes[i]
        ax.plot(time, data[col], lw=0.9)
        ax.set_title(col)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("time [s]")

    for i in range(n, len(axes)):
        axes[i].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_indexed_group(
    time: np.ndarray,
    data: Dict[str, np.ndarray],
    prefix: str,
    cols: List[str],
    out_dir: Path,
    max_subplots: int,
) -> List[Path]:
    saved: List[Path] = []
    if not cols:
        return saved

    chunks = [cols[i : i + max_subplots] for i in range(0, len(cols), max_subplots)]

    for chunk_id, chunk in enumerate(chunks, start=1):
        n = len(chunk)
        ncols = 3
        nrows = math.ceil(n / ncols)

        fig, axes = plt.subplots(nrows, ncols, figsize=(18, max(3.5 * nrows, 4)), sharex=True)
        axes = np.array(axes).reshape(-1)

        title = f"{prefix} ({len(cols)} channels)"
        if len(chunks) > 1:
            title += f" - part {chunk_id}/{len(chunks)}"
        fig.suptitle(title, fontsize=14)

        for i, col in enumerate(chunk):
            ax = axes[i]
            ax.plot(time, data[col], lw=0.9)
            ax.set_title(col)
            ax.grid(True, alpha=0.3)
            ax.set_xlabel("time [s]")

        for i in range(n, len(axes)):
            axes[i].axis("off")

        fig.tight_layout()

        filename = f"{prefix}.png" if len(chunks) == 1 else f"{prefix}_part{chunk_id}.png"
        out_path = out_dir / filename
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        saved.append(out_path)

    return saved


def main() -> None:
    args = parse_args()
    script_dir = Path(__file__).resolve().parent

    csv_path = args.csv if args.csv is not None else find_latest_csv(script_dir)
    if not csv_path.is_absolute():
        csv_path = (Path.cwd() / csv_path).resolve()

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    out_dir = args.out_dir
    if out_dir is None:
        out_dir = csv_path.parent / f"plots_{csv_path.stem}"
    elif not out_dir.is_absolute():
        out_dir = (Path.cwd() / out_dir).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading: {csv_path}")
    header, data = load_csv_numeric(csv_path)

    if "time" not in data:
        raise RuntimeError("CSV must contain a 'time' column.")

    time = data["time"]
    scalar_cols, indexed_cols = split_columns(header)

    saved_files: List[Path] = []

    scalar_path = out_dir / "scalars.png"
    plot_scalar_group(time, data, scalar_cols, scalar_path)
    if scalar_cols:
        saved_files.append(scalar_path)

    for prefix in sorted(indexed_cols.keys()):
        saved_files.extend(
            plot_indexed_group(
                time=time,
                data=data,
                prefix=prefix,
                cols=indexed_cols[prefix],
                out_dir=out_dir,
                max_subplots=max(1, args.max_subplots),
            )
        )

    print(f"[INFO] Saved {len(saved_files)} plot files to: {out_dir}")
    for p in saved_files:
        print(f"  - {p.name}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
