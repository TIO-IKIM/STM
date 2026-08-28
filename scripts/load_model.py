#!/usr/bin/env python3
"""Load an STM retriever and score a query–passage pair.

  python scripts/load_model.py --model-path ikim-uk-essen/stm_qwen --embedding-dimension 1024
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model_utils import (
    get_detailed_instruct_passage,
    get_detailed_instruct_query,
    load_decoder_model,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", default="ikim-uk-essen/stm_qwen")
    p.add_argument("--embedding-dimension", type=int, default=1024)
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--no-flash-attention", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model = load_decoder_model(
        args.model_path,
        args.embedding_dimension,
        args.max_seq_length,
        use_flash_attention=not args.no_flash_attention,
    )

    task = "Given a question, retrieve relevant passages that answer the question"
    query = get_detailed_instruct_query(task, "What are the side effects of metformin?")
    passage = get_detailed_instruct_passage(
        "Metformin can cause lactic acidosis in rare cases."
    )

    emb = model.encode([query, passage], normalize_embeddings=False)
    score = float(emb[0] @ emb[1])
    print(f"dot product: {score:.4f}")


if __name__ == "__main__":
    main()
