"""Model loading helpers for STM retrievers (last-token / EOS pooling)."""

import logging

import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Pooling, Transformer

logger = logging.getLogger(__name__)


def get_detailed_instruct_query(task_description: str, query: str) -> str:
    return f"{task_description}\nQuery: {query}"


def get_detailed_instruct_passage(passage: str) -> str:
    return f"Represent this passage\npassage: {passage}"


def _build_model(
    model_name: str,
    embedding_dimension: int,
    max_seq_length: int,
    model_kwargs: dict,
) -> SentenceTransformer:
    # model_kwargs must go through Transformer's model_args, not SentenceTransformer's.
    transformer = Transformer(
        model_name_or_path=model_name,
        tokenizer_args={"add_eos_token": True},
        max_seq_length=max_seq_length,
        model_args=model_kwargs,
    )
    pooling = Pooling(embedding_dimension, pooling_mode="lasttoken")
    model = SentenceTransformer(modules=[transformer, pooling])
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
        model.tokenizer.padding_side = "left"
    return model


def load_decoder_model(
    model_name: str,
    embedding_dimension: int,
    max_seq_length: int = 512,
    use_flash_attention: bool = True,
) -> SentenceTransformer:
    """Load a decoder-only SentenceTransformer with last-token pooling."""
    model_kwargs: dict = {}
    if torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16

    if use_flash_attention and torch.cuda.is_available():
        try:
            return _build_model(
                model_name,
                embedding_dimension,
                max_seq_length,
                {**model_kwargs, "attn_implementation": "flash_attention_2"},
            )
        except Exception:
            logger.warning(
                "flash_attention_2 unavailable for %s, falling back to sdpa.", model_name, exc_info=True
            )

    if torch.cuda.is_available():
        try:
            return _build_model(
                model_name,
                embedding_dimension,
                max_seq_length,
                {**model_kwargs, "attn_implementation": "sdpa"},
            )
        except Exception:
            logger.warning(
                "sdpa attention unavailable for %s, falling back to the library default.", model_name, exc_info=True
            )

    return _build_model(model_name, embedding_dimension, max_seq_length, model_kwargs)
