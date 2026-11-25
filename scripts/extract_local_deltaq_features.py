from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


Z_NORMALIZATION_EPSILON = 1e-12

# NumPy 2.x pickles refer to ``numpy._core`` while older environments expose
# the same modules under ``numpy.core``. Register read-only compatibility
# aliases before loading the archived Severson dictionaries.
if "numpy._core.numeric" not in sys.modules:
    import numpy.core as _np_core
    import numpy.core.numeric as _np_numeric
    import numpy.core.multiarray as _np_multiarray

    sys.modules["numpy._core"] = _np_core
    sys.modules["numpy._core.numeric"] = _np_numeric
    sys.modules["numpy._core.multiarray"] = _np_multiarray

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from extract_severson_official_features import get_cycle, load_combined_batches  # noqa: E402


def finite(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    return arr[np.isfinite(arr)]


def slope(values: np.ndarray) -> float:
    clean = finite(values)
    if clean.size < 2:
        return float("nan")
    x = np.arange(clean.size, dtype=float)
    return float(np.polyfit(x, clean, 1)[0])


def segment_stats(prefix: str, values: np.ndarray) -> dict[str, float]:
    clean = finite(values)
    if clean.size == 0:
        return {
            f"{prefix}_mean": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_var": float("nan"),
            f"{prefix}_abs_mean": float("nan"),
            f"{prefix}_min": float("nan"),
            f"{prefix}_max": float("nan"),
            f"{prefix}_range": float("nan"),
            f"{prefix}_slope": float("nan"),
            f"{prefix}_energy": float("nan"),
        }
    return {
        f"{prefix}_mean": float(np.mean(clean)),
        f"{prefix}_std": float(np.std(clean, ddof=1)) if clean.size > 1 else 0.0,
        f"{prefix}_var": float(np.var(clean, ddof=0)),
        f"{prefix}_abs_mean": float(np.mean(np.abs(clean))),
        f"{prefix}_min": float(np.min(clean)),
        f"{prefix}_max": float(np.max(clean)),
        f"{prefix}_range": float(np.max(clean) - np.min(clean)),
        f"{prefix}_slope": slope(clean),
        f"{prefix}_energy": float(np.mean(clean**2)),
    }


def z_normalize(rows: np.ndarray) -> np.ndarray:
    means = np.mean(rows, axis=1, keepdims=True)
    stds = np.std(rows, axis=1, keepdims=True)
    return (rows - means) / np.maximum(stds, Z_NORMALIZATION_EPSILON)


def matrix_profile_summary(prefix: str, values: np.ndarray, m: int, stride: int) -> dict[str, float]:
    clean = finite(values)
    if clean.size < (m + 2 * stride):
        return {f"{prefix}_{name}": float("nan") for name in [
            "mp_min",
            "mp_mean",
            "mp_std",
            "mp_q90",
            "mp_q95",
            "mp_max",
            "motif_pos_norm",
            "discord_pos_norm",
            "motif_mean",
            "motif_std",
            "motif_slope",
            "discord_mean",
            "discord_std",
            "discord_slope",
        ]}

    starts = np.arange(0, clean.size - m + 1, stride)
    subseqs = np.vstack([clean[start : start + m] for start in starts])
    zn = z_normalize(subseqs)
    diff = zn[:, None, :] - zn[None, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    exclusion = max(1, int(np.ceil(m / stride / 2)))
    for i in range(dist.shape[0]):
        low = max(0, i - exclusion)
        high = min(dist.shape[0], i + exclusion + 1)
        dist[i, low:high] = np.inf
    profile = np.min(dist, axis=1)
    profile = profile[np.isfinite(profile)]
    if profile.size == 0:
        return {f"{prefix}_mp_{name}": float("nan") for name in ["min", "mean", "std", "q90", "q95", "max"]}

    full_profile = np.min(dist, axis=1)
    valid = np.isfinite(full_profile)
    valid_indices = np.flatnonzero(valid)
    motif_idx = int(valid_indices[np.argmin(full_profile[valid])])
    discord_idx = int(valid_indices[np.argmax(full_profile[valid])])
    motif = subseqs[motif_idx]
    discord = subseqs[discord_idx]
    denom = max(float(clean.size - m), 1.0)
    return {
        f"{prefix}_mp_min": float(np.min(profile)),
        f"{prefix}_mp_mean": float(np.mean(profile)),
        f"{prefix}_mp_std": float(np.std(profile, ddof=1)) if profile.size > 1 else 0.0,
        f"{prefix}_mp_q90": float(np.quantile(profile, 0.90)),
        f"{prefix}_mp_q95": float(np.quantile(profile, 0.95)),
        f"{prefix}_mp_max": float(np.max(profile)),
        f"{prefix}_motif_pos_norm": float(starts[motif_idx] / denom),
        f"{prefix}_discord_pos_norm": float(starts[discord_idx] / denom),
        f"{prefix}_motif_mean": float(np.mean(motif)),
        f"{prefix}_motif_std": float(np.std(motif, ddof=1)) if motif.size > 1 else 0.0,
        f"{prefix}_motif_slope": slope(motif),
        f"{prefix}_discord_mean": float(np.mean(discord)),
        f"{prefix}_discord_std": float(np.std(discord, ddof=1)) if discord.size > 1 else 0.0,
        f"{prefix}_discord_slope": slope(discord),
    }


def delta_q(cell: dict[str, Any], cycle_a: int, cycle_b: int) -> np.ndarray:
    low = get_cycle(cell, cycle_a)
    high = get_cycle(cell, cycle_b)
    qd_low = np.asarray(low["Qdlin"], dtype=float).reshape(-1)
    qd_high = np.asarray(high["Qdlin"], dtype=float).reshape(-1)
    n = min(qd_low.size, qd_high.size)
    return qd_high[:n] - qd_low[:n]


def local_deltaq_features(
    cell_id: str,
    cell: dict[str, Any],
    prefix: str,
    cycle_a: int,
    cycle_b: int,
    segment_length: int,
    segment_stride: int,
    mp_length: int,
    mp_stride: int,
) -> dict[str, float | int]:
    try:
        dq = delta_q(cell, cycle_a, cycle_b)
    except KeyError:
        return {f"{prefix}_ldq_available": 0}

    row: dict[str, float | int] = {
        f"{prefix}_ldq_available": 1,
        f"{prefix}_ldq_points": int(finite(dq).size),
        f"{prefix}_ldq_cycle_a": int(cycle_a),
        f"{prefix}_ldq_cycle_b": int(cycle_b),
    }
    starts = list(range(0, max(0, dq.size - segment_length + 1), segment_stride))
    for idx, start in enumerate(starts):
        stop = start + segment_length
        seg_prefix = f"{prefix}_ldq_seg{idx:02d}_{start:04d}_{stop:04d}"
        row.update(segment_stats(seg_prefix, dq[start:stop]))
    row.update(matrix_profile_summary(f"{prefix}_ldq", dq, mp_length, mp_stride))
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-features", default="data/early_features_N150_extended_20170630.csv")
    parser.add_argument("--pkl-dir", default="data/severson_raw/pkl_20170630")
    parser.add_argument("--output", default="data/early_features_N150_local_deltaq_20170630.csv")
    parser.add_argument("--qc-output", default="data/early_features_N150_local_deltaq_qc_20170630.csv")
    parser.add_argument("--window", type=int, default=150)
    parser.add_argument("--cycle-a", type=int, default=9)
    parser.add_argument("--cycle-b", type=int, default=None)
    parser.add_argument("--segment-length", type=int, default=50)
    parser.add_argument("--segment-stride", type=int, default=25)
    parser.add_argument("--mp-length", type=int, default=50)
    parser.add_argument("--mp-stride", type=int, default=10)
    args = parser.parse_args()

    prefix = f"N{args.window}"
    cycle_b = args.cycle_b if args.cycle_b is not None else args.window - 1
    base = pd.read_csv(args.base_features)
    bat_dict, _ = load_combined_batches(Path(args.pkl_dir))

    rows: list[dict[str, object]] = []
    for _, base_row in base.iterrows():
        cell_id = str(base_row["cell_id"])
        row = base_row.to_dict()
        row.update(
            local_deltaq_features(
                cell_id=cell_id,
                cell=bat_dict[cell_id],
                prefix=prefix,
                cycle_a=args.cycle_a,
                cycle_b=cycle_b,
                segment_length=args.segment_length,
                segment_stride=args.segment_stride,
                mp_length=args.mp_length,
                mp_stride=args.mp_stride,
            )
        )
        rows.append(row)

    out = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)

    local_cols = [c for c in out.columns if "_ldq_" in c]
    qc = pd.DataFrame(
        [
            {
                "feature": col,
                "missing_count": int(out[col].isna().sum()),
                "missing_fraction": float(out[col].isna().mean()),
                "n_unique": int(out[col].nunique(dropna=True)),
            }
            for col in local_cols
        ]
    ).sort_values(["missing_count", "feature"], ascending=[False, True])
    qc_output = Path(args.qc_output)
    qc.to_csv(qc_output, index=False)

    print(f"Saved {out.shape[0]} rows, {out.shape[1]} columns -> {output}")
    print(f"Local DeltaQ columns: {len(local_cols)}")
    print(f"Saved QC -> {qc_output}")
    missing = qc[qc["missing_count"] > 0]
    if not missing.empty:
        print(missing.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
