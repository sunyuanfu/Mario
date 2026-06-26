# Mario (CVPR 2026) Open-Source Release

## Cover

## Overview

## Table of Contents

- [Repository Structure](#repository-structure)
- [Environment Setup](#environment-setup)
- [Data Policy and Expected Layout](#data-policy-and-expected-layout)
- [Running the Pipeline](#running-the-pipeline)
- [Stage 1: Feature Generation](#stage-1-feature-generation)
- [Stage 2: LLM Tuning and Evaluation](#stage-2-llm-tuning-and-evaluation)
- [Outputs](#outputs)
- [Acknowledgments](#acknowledgments)
- [License](#license)
- [Notes](#notes)

## Repository Structure

```text
Mario_CVPR2026_final_release/
├── ACKNOWLEDGMENTS.md
├── LICENSE
├── README.md
├── README_mario.md
├── template.md
├── requirements.txt
├── scripts/
│   ├── train_movies_paper_ddp.sh
│   └── train_movies_stage2_ddp.sh
├── stage1/
│   ├── main.py
│   ├── dataset_amazon_small.py
│   ├── dataset_amazon_large.py
│   ├── models.py
│   ├── config.json
│   ├── configs/
│   └── src/
└── stage2/
    ├── train_llm.py
    └── mario_llm_stage2/
```

The release is organized as a two-stage pipeline:

- `stage1/` trains the multimodal encoder and exports node-level text/image embeddings.
- `stage2/` performs modality-adaptive LLM tuning and reports the final node-classification metrics.
- `scripts/` contains end-to-end shell entry points for the Movies experiment.

Stage 1 is used for feature generation only. Final task metrics are produced by Stage 2.

## Environment Setup

Install the Python dependencies from the repository root:

```bash
pip install -r requirements.txt
```

The listed dependencies are:

```text
torch>=2.1
dgl>=1.1
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
tqdm>=4.66
transformers>=4.44
peft>=0.12
```

For multi-GPU Stage 2 training, use `torchrun`.

## Data Policy and Expected Layout

Datasets and model checkpoints are not included in this repository. Keep data and local LLM checkpoints outside the code repository, then pass their paths through environment variables or command-line arguments.

For the Movies experiment, the expected dataset layout is:

```text
<DATA_ROOT>/
└── Movies/
    ├── Movies.csv
    ├── MoviesGraph.pt
    ├── ImageFeature/
    │   └── Movies_Llama-3.2-11B-Vision-Instruct_visual.npy
    └── TextFeature/
        └── Movies_Llama_3.2_11B_Vision_Instruct_512_mean.npy
```

`Movies.csv` and `MoviesGraph.pt` are required by Stage 2. The `ImageFeature/` and `TextFeature/` files are used by Stage 1 when `--use_large_features` is enabled.

## Running the Pipeline

The simplest way to run the Movies pipeline is:

```bash
LLM_MODEL_PATH=/path/to/Llama-3.1-8B-Instruct \
DATA_ROOT=/path/to/data/MAGB \
GPUS=0,1,2,3 \
bash scripts/train_movies_stage2_ddp.sh
```

Required environment variable:

- `LLM_MODEL_PATH`: local path to the LLM checkpoint.

Optional environment variables:

- `DATA_ROOT`: dataset root. Default: `../data/MAGB`.
- `GPUS`: comma-separated GPU IDs. Default: `0,1,2,3`.
- `STAGE2_RESUME`: optional Stage 2 checkpoint path to resume from.

The script runs:

1. Stage 1 feature generation from `stage1/main.py`.
2. Stage 2 distributed LLM tuning from `stage2/train_llm.py`.

## Stage 1: Feature Generation

Stage 1 trains the encoder and exports node embeddings for Stage 2.

Run Stage 1 manually with:

```bash
cd stage1

DATA_ROOT=/path/to/data/MAGB python main.py \
  --dataset Movies \
  --use_large_features \
  --n_epochs 80 \
  --lr 0.001 \
  --lr_scheduler_gamma 0.95
```

Stage 1 writes:

- `Movies_stage1_mlp_trainonly.pth`: intermediate train-node feature snapshot.
- `Movies_stage1_mlp.pth`: all-node text/image embeddings used by Stage 2.

## Stage 2: LLM Tuning and Evaluation

Stage 2 consumes the Stage 1 embeddings, tunes the LLM components, performs validation-based checkpointing, and reports final test metrics.

Run Stage 2 manually with:

```bash
cd stage2

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 train_llm.py \
  --dataset Movies \
  --data-root /path/to/data/MAGB \
  --stage1-feature ../stage1/Movies_stage1_mlp.pth \
  --llm-model-path /path/to/Llama-3.1-8B-Instruct \
  --epochs 10 \
  --batch-size 4 \
  --inference-policy router \
  --output-dir runs/mario_movies_stage2
```

To resume from an existing Stage 2 checkpoint, add:

```bash
--resume-checkpoint /path/to/best_stage2_llm.pt
```

Supported inference policies:

- `router`
- `text`
- `image`
- `mm`
- `vote3`

## Outputs

Stage 1 outputs are saved under `stage1/`:

```text
stage1/
├── Movies_stage1_mlp_trainonly.pth
└── Movies_stage1_mlp.pth
```

Stage 2 outputs are saved under the selected `--output-dir`:

```text
stage2/runs/mario_movies_stage2/
├── best_stage2_llm.pt
├── history.jsonl
├── metrics.json
└── test_predictions.jsonl
```

The final node-classification metrics are written to:

```text
stage2/runs/mario_movies_stage2/metrics.json
```

## Acknowledgments

We sincerely thank everyone who contributed to the Mario codebase, including
the implementation, refactoring, testing, documentation, and release
preparation efforts that made this open-source release possible.

We also thank the broader open-source community for the libraries and tools
that this project builds upon.

## License

This project is released under the MIT License. Please see [LICENSE](LICENSE).

## Notes

- Stage 1 does not report final task metrics.
- Stage 2 evaluates from the best validation checkpoint.
- Datasets and model checkpoints are not included in this release.
- Release documentation is kept in English for open-source consistency.
