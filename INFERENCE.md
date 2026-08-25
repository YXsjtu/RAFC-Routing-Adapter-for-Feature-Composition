# Released checkpoint inference

Run these commands from the repository on the experiment server. The local-only
data and LWM checkpoint must be present. Commands evaluate the held-out test
split at 0, 5, 10, 15, 20, 25, and 30 dB.

```bash
cd /home/shiyuxuan/RAFC-Github
DATA=.
LWM=weights/pretrained/lwm1_1.pth
CONTRA=weights/pretrained/contrawimae.pt
CE=data_cache/channel_estimation_unseen_city9_72sc/manifest.json
CP=data_cache/channel_prediction_fixed_144sc/manifest.json
BEAM=data_cache/beam_prediction_64codebook_3p5ghz_72sc/manifest.json
```

## Channel estimation

### LWM1.1 baseline

```bash
python tasks/channel_tasks.py --task estimation --mode lwm --device cuda:0 --seed 2026 --data-root "$DATA" --manifest "$CE" --lwm-checkpoint "$LWM" --eval-only --checkpoint weights/downstream/lwm1_1/channel_estimation/baseline.pt
```

### LWM1.1 + RAFC

```bash
python tasks/channel_tasks.py --task estimation --mode lwm_rafc --device cuda:0 --seed 2026 --data-root "$DATA" --manifest "$CE" --lwm-checkpoint "$LWM" --eval-only --checkpoint weights/downstream/lwm1_1/channel_estimation/rafc.pt
```

### ContraWiMAE baseline

```bash
python tasks/channel_tasks.py --task estimation --mode contrawimae --device cuda:0 --seed 2026 --data-root "$DATA" --manifest "$CE" --contra-checkpoint "$CONTRA" --eval-only --checkpoint weights/downstream/contrawimae/channel_estimation/baseline.pt
```

### ContraWiMAE + RAFC

```bash
python tasks/channel_tasks.py --task estimation --mode contrawimae_rafc --device cuda:0 --seed 2026 --data-root "$DATA" --manifest "$CE" --contra-checkpoint "$CONTRA" --eval-only --checkpoint weights/downstream/contrawimae/channel_estimation/rafc.pt
```

## Channel prediction

### LWM1.1 baseline

```bash
python tasks/channel_tasks.py --task prediction --mode lwm --device cuda:0 --seed 2026 --data-root "$DATA" --manifest "$CP" --lwm-checkpoint "$LWM" --eval-only --checkpoint weights/downstream/lwm1_1/channel_prediction/baseline.pt
```

### LWM1.1 + RAFC

```bash
python tasks/channel_tasks.py --task prediction --mode lwm_rafc --device cuda:0 --seed 2026 --data-root "$DATA" --manifest "$CP" --lwm-checkpoint "$LWM" --eval-only --checkpoint weights/downstream/lwm1_1/channel_prediction/rafc.pt
```

### ContraWiMAE baseline

```bash
python tasks/channel_tasks.py --task prediction --mode contrawimae --device cuda:0 --seed 2026 --data-root "$DATA" --manifest "$CP" --contra-checkpoint "$CONTRA" --eval-only --checkpoint weights/downstream/contrawimae/channel_prediction/baseline.pt
```

### ContraWiMAE + RAFC

```bash
python tasks/channel_tasks.py --task prediction --mode contrawimae_rafc --device cuda:0 --seed 2026 --data-root "$DATA" --manifest "$CP" --contra-checkpoint "$CONTRA" --eval-only --checkpoint weights/downstream/contrawimae/channel_prediction/rafc.pt
```

## 64-codebook beam prediction

### LWM1.1 baseline

```bash
python tasks/beam_prediction.py --backbone lwm1_1 --variant baseline --device cuda:0 --seed 2026 --data-root "$DATA" --manifest "$BEAM" --checkpoint weights/downstream/lwm1_1/beam_prediction/baseline.pt
```

### LWM1.1 + RAFC

```bash
python tasks/beam_prediction.py --backbone lwm1_1 --variant rafc --device cuda:0 --seed 2026 --data-root "$DATA" --manifest "$BEAM" --checkpoint weights/downstream/lwm1_1/beam_prediction/rafc.pt
```

### ContraWiMAE baseline

```bash
python tasks/beam_prediction.py --backbone contrawimae --variant baseline --device cuda:0 --seed 2026 --data-root "$DATA" --manifest "$BEAM" --checkpoint weights/downstream/contrawimae/beam_prediction/baseline.pt
```

### ContraWiMAE + RAFC

```bash
python tasks/beam_prediction.py --backbone contrawimae --variant rafc --device cuda:0 --seed 2026 --data-root "$DATA" --manifest "$BEAM" --checkpoint weights/downstream/contrawimae/beam_prediction/rafc.pt
```
