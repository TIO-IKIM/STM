#!/usr/bin/env python3
"""Merge full expert models with fixed weights via mergekit-yaml.

  python scripts/merge_experts.py \\
    --config configs/examples/task_arithmetic.yaml \\
    --out outputs/merged \\
    --gpu 0
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import torch


def run_merge(config: Path, out: Path, gpu: int | None, dry_run: bool) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["mergekit-yaml", str(config), str(out), "--lazy-unpickle"]
    cmd.append("--cuda" if torch.cuda.is_available() else "--no-cuda")
    env = os.environ.copy()
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"Config: {config}")
    print(f"Out:    {out}")
    if dry_run:
        prefix = f"CUDA_VISIBLE_DEVICES={gpu} " if gpu is not None else ""
        print(f"[dry-run] {prefix}{' '.join(cmd)}")
        return 0
    return subprocess.run(cmd, env=env).returncode


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--gpu", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.config.is_file():
        print(f"Config not found: {args.config}", file=sys.stderr)
        return 1
    return run_merge(args.config.resolve(), args.out.resolve(), args.gpu, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
