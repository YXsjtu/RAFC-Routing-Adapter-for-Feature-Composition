#!/usr/bin/env python3
"""Download DeepMIMO scenarios and rebuild all three task datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SCENES = ("city_5_philadelphia", "city_8_dallas", "city_9_sanfrancisco")


def relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def ensure_scenarios(scenario_root: Path, download: bool):
    import deepmimo as dm
    from scripts.data.deepmimo_recipe import configure_scenario_root

    configure_scenario_root(scenario_root)
    missing = [scene + "_3p5" for scene in SCENES
               if not (scenario_root / (scene + "_3p5")).is_dir()]
    if missing and not download:
        raise FileNotFoundError(
            "Missing DeepMIMO scenarios: " + ", ".join(missing)
            + ". Re-run with --download."
        )
    for scenario in missing:
        dm.download(scenario, output_dir=str(scenario_root))
        if not (scenario_root / scenario).is_dir():
            raise RuntimeError(f"DeepMIMO did not create {scenario_root / scenario}")


def prepare_estimation(root: Path, overwrite: bool):
    from scripts.data.deepmimo_recipe import render_scene_to_file

    folder = root / "data_cache/channel_estimation_unseen_city9_72sc"
    path = folder / "city_9_sanfrancisco_csi.npy"
    render_scene_to_file(
        "city_9_sanfrancisco", path, 72, 14, seed=2026,
        diverse=False, overwrite=overwrite,
    )
    manifest = {
        "scene": "city_9_sanfrancisco", "path": relative(path, root),
        "samples": 6000, "shape": [6000, 4, 4, 72, 14],
        "unseen_by_pretraining": True,
        "physics": {
            "carrier_hz": 3.5e9, "bs_array": [4, 1], "ue_array": [4, 1],
            "spacing_lambda": 0.5, "paths": 25, "speed_mps": 5.0,
            "fixed_orientation": True, "subcarriers": 72,
            "subcarrier_spacing_hz": 30000.0, "ofdm_symbols": 14,
        },
        "reproduction": {"receiver_indices": "scripts/data/indices", "seed": 2026},
    }
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def prepare_prediction(root: Path, overwrite: bool):
    from scripts.data.deepmimo_recipe import render_scene_to_file

    folder = root / "data_cache/channel_prediction_fixed_144sc"
    records = []
    for scene in ("city_5_philadelphia", "city_8_dallas"):
        path = folder / f"{scene}_csi.npy"
        render_scene_to_file(
            scene, path, 144, 28, seed=2026, diverse=False,
            overwrite=overwrite,
        )
        records.append({"scene": scene, "path": relative(path, root), "samples": 6000})
    manifest = {
        "records": records,
        "physics": {
            "bs_array": [4, 1], "ue_array": [4, 1], "spacing_lambda": 0.5,
            "paths": 25, "speed_mps": 5.0, "fixed_orientation": True,
            "subcarriers": 144, "spacing_hz": 30000.0, "symbols": 28,
        },
        "split": {"seed": 2026, "train": 0.70, "validation": 0.15, "test": 0.15},
        "temporal_pair": {"history_symbols": [0, 14], "target_symbols": [14, 28]},
        "reproduction": {"receiver_indices": "scripts/data/indices", "seed": 2026},
    }
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def prepare_beam(root: Path, overwrite: bool):
    from scripts.data.deepmimo_recipe import beam_labels, dft_codebook, render_scene_to_file

    source_folder = root / "data_cache/fixed_prb_72sc_30khz"
    output = root / "data_cache/beam_prediction_64codebook_3p5ghz_72sc"
    output.mkdir(parents=True, exist_ok=True)
    codebook, spatial_frequency = dft_codebook(4, 64)
    codebook_path = output / "codebook_64.npy"
    frequency_path = output / "spatial_frequency_64.npy"
    np.save(codebook_path, codebook)
    np.save(frequency_path, spatial_frequency)
    records, all_labels = [], []
    offset = 0
    for scene in ("city_5_philadelphia", "city_8_dallas"):
        csi_path = source_folder / f"validation_{scene}_csi.npy"
        render_scene_to_file(
            scene, csi_path, 72, 14, seed=42, diverse=True,
            overwrite=overwrite,
        )
        csi = np.load(csi_path, mmap_mode="r")
        labels, best, second = beam_labels(csi, codebook)
        label_path = output / f"{scene}_labels.npy"
        best_path = output / f"{scene}_best_power.npy"
        margin_path = output / f"{scene}_power_margin.npy"
        np.save(label_path, labels)
        np.save(best_path, best)
        np.save(margin_path, (best - second).astype(np.float32))
        records.append({
            "scene": scene, "csi": relative(csi_path, root),
            "labels": relative(label_path, root), "best_power": relative(best_path, root),
            "power_margin": relative(margin_path, root), "samples": 6000, "offset": offset,
        })
        offset += len(csi)
        all_labels.append(labels)
    labels = np.concatenate(all_labels)
    permutation = np.random.default_rng(2026).permutation(len(labels))
    n_train, n_validation = int(0.70 * len(labels)), int(0.15 * len(labels))
    split_path = output / "split_indices.npz"
    np.savez(
        split_path,
        train=permutation[:n_train],
        validation=permutation[n_train:n_train + n_validation],
        test=permutation[n_train + n_validation:],
    )
    manifest = {
        "task": "same_band_beam_prediction", "carrier_hz": 3_500_000_000,
        "subcarriers": 72, "ofdm_symbols": 14, "records": records,
        "num_beams": 64, "codebook": relative(codebook_path, root),
        "codebook_type": "oversampled_half_wavelength_ula_dft",
        "spatial_frequency": relative(frequency_path, root),
        "label_rule": "argmax mean_rx_frequency_time |H w_k|^2 from full clean CSI",
        "split": {"seed": 2026, "train": 0.70, "validation": 0.15, "test": 0.15,
                  "path": relative(split_path, root)},
        "class_distribution": np.bincount(labels, minlength=64).tolist(),
        "physical_profiles": "scripts/data/pretrain_3p5_diverse.yaml",
        "reproduction": {"receiver_indices": "scripts/data/indices", "profile_seed": 42},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def verify(root: Path, tasks):
    if "estimation" in tasks:
        manifest_path = root / "data_cache/channel_estimation_unseen_city9_72sc/manifest.json"
        manifest = json.loads(manifest_path.read_text())
        array = np.load(root / manifest["path"], mmap_mode="r")
        if array.shape != (6000, 4, 4, 72, 14) or array.dtype != np.complex64:
            raise RuntimeError(f"Invalid estimation array: {array.shape}, {array.dtype}")
        print(f"[verified] {manifest_path}")
    if "prediction" in tasks:
        manifest_path = root / "data_cache/channel_prediction_fixed_144sc/manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for record in manifest["records"]:
            array = np.load(root / record["path"], mmap_mode="r")
            if array.shape != (6000, 4, 4, 144, 28) or array.dtype != np.complex64:
                raise RuntimeError(f"Invalid prediction array {record['path']}: {array.shape}")
        print(f"[verified] {manifest_path}")
    if "beam" in tasks:
        manifest_path = root / "data_cache/beam_prediction_64codebook_3p5ghz_72sc/manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest["num_beams"] != 64:
            raise RuntimeError("Beam manifest is not 64-codeword")
        total = 0
        for record in manifest["records"]:
            csi = np.load(root / record["csi"], mmap_mode="r")
            labels = np.load(root / record["labels"], mmap_mode="r")
            if csi.shape != (6000, 4, 4, 72, 14) or csi.dtype != np.complex64:
                raise RuntimeError(f"Invalid beam CSI {record['csi']}: {csi.shape}, {csi.dtype}")
            if labels.shape != (6000,) or labels.min() < 0 or labels.max() >= 64:
                raise RuntimeError(f"Invalid beam labels: {record['labels']}")
            total += len(labels)
        codebook = np.load(root / manifest["codebook"])
        if codebook.shape != (4, 64) or codebook.dtype != np.complex64:
            raise RuntimeError(f"Invalid beam codebook: {codebook.shape}, {codebook.dtype}")
        split = np.load(root / manifest["split"]["path"])
        sizes = tuple(map(len, (split["train"], split["validation"], split["test"])))
        if sizes != (8400, 1800, 1800):
            raise RuntimeError(f"Invalid beam split: {sizes}")
        combined = np.concatenate([split["train"], split["validation"], split["test"]])
        if total != 12000 or not np.array_equal(np.sort(combined), np.arange(total)):
            raise RuntimeError("Beam split does not cover every sample exactly once")
        print(f"[verified] {manifest_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", choices=("all", "estimation", "prediction", "beam"), default=["all"])
    parser.add_argument("--scenario-root", type=Path, default=PROJECT_ROOT / "deepmimo_scenarios")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    tasks = {"estimation", "prediction", "beam"} if "all" in args.tasks else set(args.tasks)
    if not args.verify_only:
        ensure_scenarios(args.scenario_root, args.download)
        if "estimation" in tasks:
            prepare_estimation(args.output_root, args.overwrite)
        if "prediction" in tasks:
            prepare_prediction(args.output_root, args.overwrite)
        if "beam" in tasks:
            prepare_beam(args.output_root, args.overwrite)
    verify(args.output_root, tasks)


if __name__ == "__main__":
    main()
