"""Unified entry point for released RAFC experiments."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BACKBONES = ("lwm1_1", "contrawimae")
TASKS = ("channel_estimation", "channel_prediction", "beam_prediction", "localization")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--backbone", choices=BACKBONES)
    parser.add_argument("--variant", choices=("baseline", "rafc"), default="rafc")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("extra_args", nargs=argparse.REMAINDER)
    return parser.parse_args()


def print_supported():
    for task in TASKS:
        for backbone in BACKBONES:
            print(f"{task:20s} {backbone:12s} layers=0,5,11  train/evaluate")


def command_for(args, seed):
    mode_prefix = "lwm" if args.backbone == "lwm1_1" else "contrawimae"
    mode = mode_prefix + ("_rafc" if args.variant == "rafc" else "")
    if args.task in ("channel_estimation", "channel_prediction"):
        if args.data_root is None or args.manifest is None:
            raise SystemExit("--data-root and --manifest are required for channel tasks")
        task = "estimation" if args.task == "channel_estimation" else "prediction"
        command = [
            sys.executable, str(ROOT / "tasks/channel_tasks.py"),
            "--task", task, "--mode", mode, "--device", args.device,
            "--seed", str(seed), "--data-root", str(args.data_root),
            "--manifest", str(args.manifest), "--run-name",
            f"{args.task}_{args.backbone}_{args.variant}_seed{seed}",
        ]
    elif args.task == "localization":
        if args.cache is None:
            raise SystemExit("--cache is required for localization")
        localization_mode = "lwm11" if args.backbone == "lwm1_1" else "contrawimae"
        if args.variant == "rafc":
            localization_mode += "_rafc"
        command = [
            sys.executable, str(ROOT / "tasks/localization.py"),
            "--mode", localization_mode, "--device", args.device,
            "--seed", str(seed), "--cache", str(args.cache), "--output",
            str(ROOT / "runs" / f"localization_{args.backbone}_{args.variant}_seed{seed}"),
        ]
    else:
        if args.data_root is None or args.manifest is None:
            raise SystemExit("--data-root and --manifest are required for beam prediction")
        command = [
            sys.executable, str(ROOT / "tasks/beam_prediction.py"),
            "--train", "--backbone", args.backbone, "--variant", args.variant,
            "--device", args.device, "--seed", str(seed),
            "--data-root", str(args.data_root), "--manifest", str(args.manifest),
            "--output-dir", str(
                ROOT / "runs" / f"beam_prediction_{args.backbone}_{args.variant}_seed{seed}"
            ),
        ]
    forwarded = args.extra_args[1:] if args.extra_args[:1] == ["--"] else args.extra_args
    return command + forwarded


def main():
    args = parse_args()
    if args.list:
        print_supported()
        return
    if args.task is None or args.backbone is None:
        raise SystemExit("--task and --backbone are required unless --list is used")
    if len(args.seeds) != len(set(args.seeds)):
        raise SystemExit("--seeds must not contain duplicates")
    for seed in args.seeds:
        command = command_for(args, seed)
        print("RUN", " ".join(map(str, command)), flush=True)
        if args.dry_run:
            continue
        completed = subprocess.run(command, cwd=ROOT, check=False)
        if completed.returncode:
            raise SystemExit(f"Experiment failed with exit code {completed.returncode}")


if __name__ == "__main__":
    main()
