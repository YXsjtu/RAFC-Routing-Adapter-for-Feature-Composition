# RAFC

RAFC (Representation-Adaptive Feature Combination) is a lightweight module
for adapting frozen wireless foundation models to downstream tasks. This
repository supports two backbones, LWM1.1 and ContraWiMAE, on three tasks:
channel estimation, channel prediction, and 64-codeword beam prediction.

## Citation

arXiv: [2606.10277](https://arxiv.org/abs/2606.10277)

First-author contact: [yuxuanshi22@gmail.com](mailto:yuxuanshi22@gmail.com)

## How RAFC works

A frozen backbone produces token representations at encoder layers 0, 5, and
11. For each input sample, RAFC pools the tokens from each selected layer,
concatenates the three pooled vectors, and predicts three routing logits with a
small two-layer MLP:

```text
s_l = MeanPool(H_l),                     l in {0, 5, 11}
alpha = Softmax(MLP([s_0; s_5; s_11]))
H_fused = alpha_0 H_0 + alpha_5 H_5 + alpha_11 H_11
prediction = TaskHead(H_fused)
```

The routing weights `alpha` are sample-dependent. The backbone remains frozen;
only the RAFC router and downstream task head are trained. A baseline uses the
last backbone representation directly and trains only the task head.

## Repository layout

```text
RAFC-Github/
├── models/                      # RAFC, LWM1.1, and ContraWiMAE definitions
├── tasks/
│   ├── channel_tasks.py         # Channel estimation/prediction train and eval
│   └── beam_prediction.py       # 64-codeword beam train and eval
├── scripts/data/                # DeepMIMO download and generation recipe
├── configs/                     # Task data and checkpoint configurations
├── weights/
│   ├── pretrained/              # Backbone checkpoints
│   └── downstream/              # Baseline and RAFC task checkpoints
├── run_experiment.py            # Unified training launcher
├── requirements.txt
└── requirements-data.txt
```

## Roadmap

WirelessGPT comparisons and localization-task support will be added in a future
release.

## 1. Download and install

Clone the repository using its GitHub URL:

```bash
git clone https://github.com/YXsjtu/RAFC-Routing-Adapter-for-Feature-Composition.git
cd RAFC-Routing-Adapter-for-Feature-Composition
```

Create or activate a Python environment and install the runtime dependencies:

```bash
conda create -n rafc python=3.12 -y
conda activate rafc
pip install -r requirements.txt
```

The released code was tested with Python 3.12, NumPy 2.0.1, PyTorch 2.5.1,
and CUDA GPUs.

The ContraWiMAE pretrained checkpoint is located at:

```text
weights/pretrained/contrawimae.pt
```

Obtain an authorized LWM1.1 checkpoint separately and place it at:

```text
weights/pretrained/lwm1_1.pth
```

## 2. Generate the DeepMIMO datasets

The repository does not upload raw DeepMIMO scenarios or generated CSI. It
contains the scenario download code, fixed receiver indices, physical channel
configuration, 64-codeword label generator, and train/validation/test split
logic required by the three tasks.

Install the data-generation dependencies:

```bash
pip install -r requirements-data.txt
```

Download the three required DeepMIMO scenarios and generate every task dataset:

```bash
python scripts/data/prepare_deepmimo_data.py \
  --download \
  --tasks all
```

The command downloads the following scenarios:

- `city_5_philadelphia_3p5`
- `city_8_dallas_3p5`
- `city_9_sanfrancisco_3p5`

It creates the following ignored local directories:

| Task | Generated manifest | Data layout |
| --- | --- | --- |
| Channel estimation | `data_cache/channel_estimation_unseen_city9_72sc/manifest.json` | `6000 × 4 × 4 × 72 × 14` complex CSI |
| Channel prediction | `data_cache/channel_prediction_fixed_144sc/manifest.json` | two `6000 × 4 × 4 × 144 × 28` arrays; the first 14 symbols predict the next 14 |
| Beam prediction | `data_cache/beam_prediction_64codebook_3p5ghz_72sc/manifest.json` | 12000 CSI samples, 64-codeword DFT labels, and an 8400/1800/1800 split |

Generate only one task when needed:

```bash
python scripts/data/prepare_deepmimo_data.py --download --tasks estimation
python scripts/data/prepare_deepmimo_data.py --download --tasks prediction
python scripts/data/prepare_deepmimo_data.py --download --tasks beam
```

Verify existing generated data without regenerating it:

```bash
python scripts/data/prepare_deepmimo_data.py --verify-only --tasks all
```

Downloaded scenarios are stored in `deepmimo_scenarios/`; generated task data
are stored in `data_cache/`. Both directories are excluded by `.gitignore`.

## 3. Downstream task scripts

Run all commands below from the repository root. Define the common paths once:

```bash
DATA=.
CE=data_cache/channel_estimation_unseen_city9_72sc/manifest.json
CP=data_cache/channel_prediction_fixed_144sc/manifest.json
BEAM=data_cache/beam_prediction_64codebook_3p5ghz_72sc/manifest.json
```

Channel-task modes are:

| Backbone | Baseline mode | RAFC mode |
| --- | --- | --- |
| LWM1.1 | `lwm` | `lwm_rafc` |
| ContraWiMAE | `contrawimae` | `contrawimae_rafc` |

### Channel estimation inference

```bash
python tasks/channel_tasks.py \
  --task estimation \
  --mode lwm_rafc \
  --device cuda:0 \
  --seed 2026 \
  --data-root "$DATA" \
  --manifest "$CE" \
  --eval-only \
  --checkpoint weights/downstream/lwm1_1/channel_estimation/rafc.pt
```

For ContraWiMAE + RAFC, use `--mode contrawimae_rafc` and checkpoint
`weights/downstream/contrawimae/channel_estimation/rafc.pt`. Replace `rafc` with
`baseline` in the checkpoint path and use the corresponding baseline mode to
evaluate a baseline.

### Channel prediction inference

```bash
python tasks/channel_tasks.py \
  --task prediction \
  --mode contrawimae_rafc \
  --device cuda:0 \
  --seed 2026 \
  --data-root "$DATA" \
  --manifest "$CP" \
  --eval-only \
  --checkpoint weights/downstream/contrawimae/channel_prediction/rafc.pt
```

### 64-codeword beam-prediction inference

```bash
python tasks/beam_prediction.py \
  --backbone lwm1_1 \
  --variant rafc \
  --device cuda:0 \
  --seed 2026 \
  --data-root "$DATA" \
  --manifest "$BEAM" \
  --checkpoint weights/downstream/lwm1_1/beam_prediction/rafc.pt
```

The beam script evaluates 0, 5, 10, 15, 20, 25, and 30 dB by default. To
evaluate only one operating point, add for example `--snrs 15`.

## 4. Train downstream models

The backbone is frozen in every downstream experiment. Baseline training
updates the task head; RAFC training updates the task head and router. Training
SNR is sampled uniformly from 0 to 30 dB.

List supported task/backbone combinations:

```bash
python run_experiment.py --list
```

### Train channel estimation

```bash
python run_experiment.py \
  --task channel_estimation \
  --backbone lwm1_1 \
  --variant rafc \
  --seeds 2026 \
  --device cuda:0 \
  --data-root "$DATA" \
  --manifest "$CE" \
  -- --epochs 1000 --patience 30 \
     --train-batch-size 32 --eval-batch-size 64
```

### Train channel prediction

```bash
python run_experiment.py \
  --task channel_prediction \
  --backbone contrawimae \
  --variant rafc \
  --seeds 2026 \
  --device cuda:0 \
  --data-root "$DATA" \
  --manifest "$CP" \
  -- --epochs 1000 --patience 30 \
     --train-batch-size 32 --eval-batch-size 64
```

### Train 64-codeword beam prediction

```bash
python run_experiment.py \
  --task beam_prediction \
  --backbone lwm1_1 \
  --variant rafc \
  --seeds 2026 \
  --device cuda:0 \
  --data-root "$DATA" \
  --manifest "$BEAM" \
  -- --epochs 500 --patience 30 \
     --train-batch-size 64 --batch-size 128
```

Use `--backbone contrawimae` to train ContraWiMAE. Use `--variant baseline` to
train without RAFC. Arguments after the standalone `--` are forwarded to the
task script.

The beam trainer can also be called directly:

```bash
python tasks/beam_prediction.py \
  --train \
  --backbone contrawimae \
  --variant rafc \
  --device cuda:0 \
  --seed 2026 \
  --data-root "$DATA" \
  --manifest "$BEAM" \
  --epochs 500 \
  --patience 30 \
  --output-dir runs/beam_prediction_contrawimae_rafc_seed2026
```

Add `--resume` to continue from `last.pt`. Channel-task training selects the
best checkpoint by validation NMSE; beam training selects it by mean validation
macro-F1 across the configured SNRs. Each run writes `best.pt`, `last.pt`, and
`results.json` under `runs/`.

## LWM1.1 license notice

The public LWM1.1 repository does not currently provide an explicit software or
checkpoint license. Therefore, the official LWM1.1 pretrained checkpoint is not
redistributed here. Users must obtain it from the upstream authors and confirm
that their intended use is permitted. See `THIRD_PARTY_NOTICES.md` for details.

## License

RAFC integration and task code are released under the root MIT license.
Third-party models, checkpoints, and datasets remain subject to their own terms.
