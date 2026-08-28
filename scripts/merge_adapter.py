#!/usr/bin/env python3
"""Merge a LoRA / PEFT adapter into base weights and save a full model.

  python scripts/merge_adapter.py \\
    --adapter /path/to/lora_checkpoint \\
    --embedding-dimension 1024 \\
    --out outputs/expert-qwen-real-medical
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model_utils import load_decoder_model


def resolve_adapter_path(adapter: Path) -> Path:
    final = adapter / "final"
    return final if final.is_dir() else adapter


def merge_adapter(
    adapter: Path,
    embedding_dimension: int,
    out: Path,
    max_seq_length: int,
    use_flash_attention: bool,
) -> None:
    load_path = resolve_adapter_path(adapter)
    print(f"Loading adapter: {load_path}")
    model = load_decoder_model(
        model_name=str(load_path),
        embedding_dimension=embedding_dimension,
        max_seq_length=max_seq_length,
        use_flash_attention=use_flash_attention,
    )

    print("Merging LoRA into base weights...")
    model[0].auto_model = model[0].auto_model.merge_and_unload()
    # Reset the internal PEFT flag so save() below writes full weights, not an adapter config.
    model[0].auto_model._hf_peft_config_loaded = False

    out.mkdir(parents=True, exist_ok=True)
    print(f"Saving → {out}")
    model.save(str(out))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--embedding-dimension", type=int, required=True)
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--no-flash-attention", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not args.adapter.exists():
        raise SystemExit(f"Adapter path not found: {args.adapter}")
    merge_adapter(
        adapter=args.adapter,
        embedding_dimension=args.embedding_dimension,
        out=args.out,
        max_seq_length=args.max_seq_length,
        use_flash_attention=not args.no_flash_attention,
    )


if __name__ == "__main__":
    main()
