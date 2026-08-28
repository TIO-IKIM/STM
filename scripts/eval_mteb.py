#!/usr/bin/env python3
"""Evaluate an STM checkpoint on MTEB retrieval tasks.

  python scripts/eval_mteb.py \\
    --model-path ikim-uk-essen/stm_qwen \\
    --embedding-dimension 1024 \\
    --batch-size 128 \\
    --benchmarks general,medical \\
    --output-dir results/stm_qwen
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import mteb
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model_utils import (
    get_detailed_instruct_passage,
    get_detailed_instruct_query,
    load_decoder_model,
)


class MTEBModel:
    """Applies STM query/passage instruction prefixes for MTEB."""

    def __init__(
        self,
        model_path: str,
        embedding_dimension: int,
        max_seq_length: int,
        datasets_config: dict,
        use_flash_attention: bool,
    ) -> None:
        self.model = load_decoder_model(
            model_path,
            embedding_dimension,
            max_seq_length,
            use_flash_attention=use_flash_attention,
        )
        self.datasets_config = datasets_config

    def _task_description(self, task_name: str) -> str:
        for group in self.datasets_config.values():
            if not isinstance(group, dict):
                continue
            section = group.get("mteb") or {}
            if task_name in section:
                return (section[task_name] or {}).get("task_description", "")
        return ""

    @torch.no_grad()
    def encode(self, sentences, task_name: str = "", prompt_type=None, batch_size: int = 64, **kwargs):
        from mteb.encoder_interface import PromptType

        self.model.eval()
        desc = self._task_description(task_name)
        if prompt_type == PromptType.query:
            sentences = [get_detailed_instruct_query(desc, s) for s in sentences]
        elif prompt_type == PromptType.passage:
            sentences = [get_detailed_instruct_passage(s) for s in sentences]
        kwargs.setdefault("show_progress_bar", True)
        kwargs.setdefault("convert_to_tensor", True)
        return self.model.encode(
            sentences,
            batch_size=batch_size,
            **kwargs,
        )


def collect_tasks(config: dict, benchmarks: list[str]) -> list[str]:
    names: list[str] = []
    for bm in benchmarks:
        section = (config.get(bm) or {}).get("mteb") or {}
        names.extend(section.keys())
    return names


def with_oom_retry(fn, logger: logging.Logger, batch_size: int):
    bs = batch_size
    while bs >= 1:
        try:
            return fn(batch_size=bs)
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e):
                raise
            torch.cuda.empty_cache()
            if bs == 1:
                raise
            logger.warning("CUDA OOM at batch_size=%s, retrying with %s", bs, bs // 2)
            bs //= 2


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", required=True)
    p.add_argument("--embedding-dimension", type=int, required=True)
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--corpus-chunk-size", type=int, default=512)
    p.add_argument("--benchmarks", default="general,medical")
    p.add_argument("--config-file", type=Path, default=ROOT / "configs" / "eval_datasets.yaml")
    p.add_argument("--output-dir", type=Path, default=ROOT / "results")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--no-flash-attention", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    else:
        print(
            f"CUDA_VISIBLE_DEVICES is already set to {os.environ['CUDA_VISIBLE_DEVICES']!r} in the "
            f"environment; --gpu {args.gpu} is ignored. Unset it first if you want --gpu to take effect.",
            file=sys.stderr,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(args.output_dir / f"eval_{datetime.now():%Y%m%d_%H%M%S}.log"),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger("eval_mteb")

    with args.config_file.open() as f:
        config = yaml.safe_load(f)

    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    task_names = collect_tasks(config, benchmarks)
    logger.info("Model: %s", args.model_path)
    logger.info("Tasks (%d): %s", len(task_names), task_names)

    model = MTEBModel(
        model_path=args.model_path,
        embedding_dimension=args.embedding_dimension,
        max_seq_length=args.max_seq_length,
        datasets_config=config,
        use_flash_attention=not args.no_flash_attention,
    )

    tasks = []
    for name in task_names:
        if name == "CUREv1":
            tasks.append(mteb.get_task(name, hf_subsets=["en"]))
        elif name == "PublicHealthQA":
            tasks.append(mteb.get_task(name, hf_subsets=["english"]))
        else:
            tasks.append(mteb.get_task(name))

    safe = args.model_path.replace("/", "_")
    out = args.output_dir / "mteb_results" / safe

    def _run(batch_size: int):
        evaluation = mteb.MTEB(tasks)
        # eval_splits left unset: not every task uses "test" (CUREv1, Nano-BEIR tasks don't).
        evaluation.run(
            model,
            output_folder=str(out),
            corpus_chunk_size=args.corpus_chunk_size,
            encode_kwargs={"batch_size": batch_size},
        )

    with_oom_retry(_run, logger, args.batch_size)
    logger.info("Results → %s", out)


if __name__ == "__main__":
    main()
