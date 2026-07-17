from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import h5py
import numpy as np


DEFAULT_MAT_FILES = {
    "batch1": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "batch2": "2017-06-30_batchdata_updated_struct_errorcorrect.mat",
    "batch3": "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}

SUMMARY_FIELDS = {
    "IR": "IR",
    "QC": "QCharge",
    "QD": "QDischarge",
    "Tavg": "Tavg",
    "Tmin": "Tmin",
    "Tmax": "Tmax",
    "chargetime": "chargetime",
    "cycle": "cycle",
}

CYCLE_FIELDS = {
    "I": "I",
    "Qc": "Qc",
    "Qd": "Qd",
    "Qdlin": "Qdlin",
    "T": "T",
    "Tdlin": "Tdlin",
    "V": "V",
    "dQdV": "discharge_dQdV",
    "t": "t",
}


def deref_array(handle: h5py.File, ref: Any) -> np.ndarray:
    return np.asarray(handle[ref][()]).reshape(-1)


def decode_policy(handle: h5py.File, ref: Any) -> str:
    raw = np.asarray(handle[ref][()])
    text = raw.tobytes()[::2].decode("utf-8", errors="ignore")
    return text.replace("\x00", "").strip()


def build_one_batch(mat_path: Path, key_prefix: str) -> dict[str, dict[str, Any]]:
    print(f"Reading {mat_path}")
    with h5py.File(mat_path, "r") as handle:
        batch = handle["batch"]
        num_cells = batch["summary"].shape[0]
        bat_dict: dict[str, dict[str, Any]] = {}

        for cell_idx in range(num_cells):
            if cell_idx % 5 == 0:
                print(f"  {key_prefix}: cell {cell_idx + 1}/{num_cells}")

            cycle_life = deref_array(handle, batch["cycle_life"][cell_idx, 0])
            policy = decode_policy(handle, batch["policy_readable"][cell_idx, 0])

            summary_ref = batch["summary"][cell_idx, 0]
            summary_group = handle[summary_ref]
            summary = {
                output_key: np.asarray(summary_group[source_key][()]).reshape(-1)
                for output_key, source_key in SUMMARY_FIELDS.items()
            }

            cycles_ref = batch["cycles"][cell_idx, 0]
            cycles_group = handle[cycles_ref]
            cycle_dict: dict[str, dict[str, np.ndarray]] = {}
            n_cycles = cycles_group["I"].shape[0]
            for cycle_idx in range(n_cycles):
                cycle_dict[str(cycle_idx)] = {
                    output_key: deref_array(handle, cycles_group[source_key][cycle_idx, 0])
                    for output_key, source_key in CYCLE_FIELDS.items()
                }

            bat_dict[f"{key_prefix}c{cell_idx}"] = {
                "cycle_life": cycle_life,
                "charge_policy": policy,
                "summary": summary,
                "cycles": cycle_dict,
            }

        return bat_dict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="data/severson_raw/pkl")
    parser.add_argument("--batch1-file", default=DEFAULT_MAT_FILES["batch1"])
    parser.add_argument("--batch2-file", default=DEFAULT_MAT_FILES["batch2"])
    parser.add_argument("--batch3-file", default=DEFAULT_MAT_FILES["batch3"])
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mat_files = {
        "batch1": args.batch1_file,
        "batch2": args.batch2_file,
        "batch3": args.batch3_file,
    }
    for batch_name, file_name in mat_files.items():
        mat_path = data_dir / file_name
        if not mat_path.exists():
            raise FileNotFoundError(f"Missing {mat_path}")
        key_prefix = batch_name.replace("batch", "b")
        bat_dict = build_one_batch(mat_path, key_prefix)
        output_path = output_dir / f"{batch_name}.pkl"
        with open(output_path, "wb") as fp:
            pickle.dump(bat_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved {len(bat_dict)} cells to {output_path}")


if __name__ == "__main__":
    main()
