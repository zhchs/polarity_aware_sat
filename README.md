# PASAT

This repository contains code provided for peer review. 

The full source code will be made publicly available upon the paper's acceptance.

## Environment Setup

Install the required dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Data Preparation

Please follow the instructions at [G4SATBench GitHub](https://github.com/zhaoyu-li/G4SATBench) to generate the SAT instances.

The generated data should be organized in the following directory structure (using the `sr` dataset as an example):

```text
<your-data-dir>
├── easy
│   └── sr
│       ├── train
│       │   ├── sat
│       │   └── unsat (contains .cnf, .core, .proof, _core_variable.pkl)
│       ├── valid
│       └── test
├── medium
└── hard
```

## Running the Code

Navigate to the `python_torch_release` directory and use `train_core.py` to start training.

### Train PASAT (Proposed Model)

To train the PASAT model on the SR Easy dataset (replacing <data-dir> with your local data directory):

```bash
python train_core.py --data_type SATBench --num_workers 8 --in_memory \
  --graph_build_mode dynamic \
  --train_dir <data-dir>/easy/sr/train \
  --train_splits unsat \
  --valid_dir <data-dir>/easy/sr/valid \
  --valid_splits unsat \
  --test_dir <data-dir>/easy/sr/test \
  --test_splits unsat \
  --dual \
  --model PASAT --v_act_fn --debug \
  --n_rounds 4 \
  --epochs 100 --lr 1e-4 --weight_decay 5e-4 --dropout 0 --clip_norm_val 1 \
  --scheduler_type exp --lr_gamma 0.871 --scheduler_steps 10 \
  --consistency_T 1 --consistency_loss_lambda 0.1 --consistency_type logits-center-mse \
  --tilde_consistency --decomp_loss_lambda 0.15 \
  --batch_size 200 --accum_step 1 --report_step 5000 --no_eval_in_report
```

### Train NeuroCore (Baseline)

To train the baseline NeuroCore model:

```bash
python train_core.py --data_type SATBench --num_workers 8 --in_memory \
  --train_dir <data-dir>/easy/sr/train \
  --train_splits unsat \
  --valid_dir <data-dir>/easy/sr/valid \
  --valid_splits unsat \
  --test_dir <data-dir>/easy/sr/test \
  --test_splits unsat \
  --model neurocore --graph LCG --v_act_fn \
  --shardshuffle 50000 --debug \
  --epochs 200 --lr 1e-4 --weight_decay 5e-4 --dropout 0 --clip_norm_val 1 \
  --scheduler_type exp --lr_gamma 0.871 --scheduler_steps 10 \
  --batch_size 100 --report_step 5000 --no_eval_in_report
```


