# Mario (CVPR 2026) Open-Source Release

This repository contains a release-ready two-stage Mario pipeline:

- `stage1`: graph-conditioned multimodal representation learning
- `stage2`: paper-faithful modality-adaptive LLM tuning and evaluation

The code is organized so that **Stage 1 does not report final task metrics**.
Only **Stage 2** reports the final node-classification metrics.
Paper-to-code alignment notes are documented in `REVIEW_PAPER_ALIGNMENT.md`.

## Repository Layout

```text
Mario_CVPR2026_open_source/
  stage1/
    main.py
    dataset_amazon_small.py
    dataset_amazon_large.py
    models.py
    src/
  stage2/
    train_llm.py
  scripts/
    train_movies_paper_ddp.sh
    train_movies_stage2_ddp.sh
  requirements.txt
  .gitignore
```

## Environment

Install dependencies:

```bash
pip install -r requirements.txt
```

For multi-GPU Stage 2, run with `torchrun`.

## Data Policy

- This repository does **not** include datasets.
- Keep datasets outside the repository and pass paths via arguments.
- Default expected Stage 2 layout:

```text
<DATA_ROOT>/Movies/
  Movies.csv
  MoviesGraph.pt
```

## Stage 1 (Feature Generation Only)

Stage 1 trains the graph-conditioned encoder and exports embeddings:

```bash
cd stage1
DATA_ROOT=/path/to/data python main.py \
  --dataset Movies \
  --use_large_features \
  --n_epochs 80 \
  --lr 0.001 \
  --lr_scheduler_gamma 0.95
```

Outputs:

- `Movies_stage1_mlp_trainonly.pth` (intermediate snapshot)
- `Movies_stage1_mlp.pth` (all-node embeddings for Stage 2)

## Stage 2 (Final Metrics)

Stage 2 performs modality-adaptive routing and LLM tuning, then reports final metrics:

```bash
cd stage2
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_llm.py \
  --dataset Movies \
  --data-root /path/to/data/MAGB \
  --stage1-feature ../stage1/Movies_stage1_mlp.pth \
  --llm-model-path /path/to/Llama-3.1-8B-Instruct \
  --resume-checkpoint runs/paper_llm_movies_ddp_exact_full_20260618_1348/best_stage2_llm.pt \
  --epochs 10 \
  --batch-size 4 \
  --inference-policy router \
  --output-dir runs/mario_movies_stage2
```

Final metrics are written to:

- `runs/.../metrics.json`

## One-Command Pipeline

Use:

```bash
bash scripts/train_movies_stage2_ddp.sh
```

Required environment variable:

- `LLM_MODEL_PATH` (local path to the LLM checkpoint)

Optional environment variables:

- `DATA_ROOT` (default: `../data/MAGB`)
- `GPUS` (default: `0,1,2,3`)

## Notes

- Stage 2 evaluation is performed from the best validation checkpoint.
- Comments and release docs are maintained in English for open-source consistency.
