# Modular Expert Merging for Biomedical Retrieval

> [Paper](https://arxiv.org/abs/2602.04731) · [Models & data](https://huggingface.co/collections/ikim-uk-essen/stm)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.10+, CUDA GPU recommended.


## Merge a LoRA adapter into full weights

This step only applies if you're starting from a LoRA adapter rather than full checkpoints.

```bash
python scripts/merge_adapter.py \
  --adapter path/to/your/lora-adapter \
  --embedding-dimension 1024 \
  --out outputs/expert-qwen-full
```



## Merge experts (mergekit)

Edit paths/weights in `configs/examples/`, then:

```bash
python scripts/merge_experts.py \
  --config configs/examples/task_arithmetic.yaml \
  --out outputs/merged_task_arithmetic \
  --gpu 0
```


| Method          | Example                                 | Needs `base_model` |
| --------------- | --------------------------------------- | ------------------ |
| Linear          | `configs/examples/linear.yaml`          | no                 |
| Task Arithmetic | `configs/examples/task_arithmetic.yaml` | yes                |
| TIES            | `configs/examples/ties.yaml`            | yes                |
| DARE-TIES       | `configs/examples/dare_ties.yaml`       | yes                |




## Load a model

```python
from src.model_utils import load_decoder_model, get_detailed_instruct_query, get_detailed_instruct_passage

model = load_decoder_model("ikim-uk-essen/stm_qwen", embedding_dimension=1024)
task = "Given a question, retrieve relevant passages that answer the question"
q = get_detailed_instruct_query(task, "What are the side effects of metformin?")
p = get_detailed_instruct_passage("Metformin can cause lactic acidosis in rare cases.")
emb = model.encode([q, p], normalize_embeddings=False)
score = float(emb[0] @ emb[1])
```



## Evaluate on MTEB

```bash
python scripts/eval_mteb.py \
  --model-path ikim-uk-essen/stm_qwen \
  --embedding-dimension 1024 \
  --batch-size 128 \
  --benchmarks general,medical \
  --output-dir results/stm_qwen
```

Tasks/prompts: `configs/eval_datasets.yaml`.

> NOTE:
> To train your own experts: [sentence-transformers' PEFT training](https://sbert.net/examples/sentence_transformer/training/peft/README.html). Please refer to the paper for implementation details. 



## Citation

```bibtex
@misc{khattab2026modularexpertmergingbiomedical,
  title         = {Modular Expert Merging for Biomedical Retrieval},
  author        = {Sameh Khattab and Jean-Philippe Corbeil and Osman Alperen {\c{C}}inar-Kora{\c{s}} and Amin Dada and Julian Friedrich and Jiawei He and Douglas Teodoro and Jens Kleesiek},
  year          = {2026},
  eprint        = {2602.04731},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2602.04731}
}
```

