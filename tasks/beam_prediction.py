"""Train or evaluate LWM1.1/ContraWiMAE 64-codebook beam predictors."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from bisect import bisect_right
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.contrawimae import ContraWiMAE
from models.lwm1_1 import LWM


LAYER_IDS = (0, 5, 11)
NUM_CLASSES = 64
PILOT_COMB = 4
SYMBOL_INDEX = 0


class PositionalEncoding(nn.Module):
    def __init__(self, dimension: int, max_len: int = 1024):
        super().__init__()
        encoding = torch.zeros(max_len, dimension)
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, dimension, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / dimension)
        )
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("pe", encoding.unsqueeze(0))

    def forward(self, value):
        return value + self.pe[:, : value.shape[1]]


class DynamicFeatureFusion(nn.Module):
    """Checkpoint-compatible RAFC router used by the beam runs."""

    def __init__(self, dimension: int):
        super().__init__()
        hidden = max(dimension // 4, 1)
        self.mlp = nn.Sequential(
            nn.Linear(3 * dimension, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 3),
        )

    def forward(self, features):
        values = list(features.values())
        alpha = self.mlp(torch.cat([value.mean(1) for value in values], -1)).softmax(-1)
        fused = sum(alpha[:, index, None, None] * value for index, value in enumerate(values))
        return fused, alpha


class BeamDataset(Dataset):
    def __init__(self, data_root: Path, records):
        self.arrays = [np.load(data_root / row["csi"], mmap_mode="r") for row in records]
        self.labels = [np.load(data_root / row["labels"], mmap_mode="r") for row in records]
        self.ends = np.cumsum([len(array) for array in self.arrays]).tolist()

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        part = bisect_right(self.ends, index)
        local = index - (self.ends[part - 1] if part else 0)
        sample = np.asarray(self.arrays[part][local])
        csi = torch.from_numpy(np.array(sample[..., SYMBOL_INDEX], copy=True))
        return csi, int(self.labels[part][local])


def sparse_observation(clean, snr_db, generator):
    mask = torch.zeros(
        (1, 1, clean.shape[2], clean.shape[3]), dtype=torch.bool, device=clean.device
    )
    for transmitter in range(clean.shape[2]):
        mask[0, 0, transmitter, transmitter::PILOT_COMB] = True
    mask = mask.expand_as(clean)
    mask_float = mask.float()
    dimensions = tuple(range(1, clean.ndim))
    count = mask_float.sum(dimensions, keepdim=True).clamp_min(1.0)
    power = (clean.abs().square() * mask_float).sum(dimensions, keepdim=True) / count
    snr = torch.as_tensor(snr_db, device=clean.device, dtype=clean.real.dtype)
    if snr.ndim == 1:
        snr = snr.reshape(clean.shape[0], *([1] * (clean.ndim - 1)))
    noise_power = power / torch.pow(10.0, snr / 10.0)
    real = torch.randn(clean.shape, device=clean.device, generator=generator)
    imaginary = torch.randn(clean.shape, device=clean.device, generator=generator)
    noisy = clean + torch.complex(real, imaginary) * torch.sqrt(noise_power / 2.0)
    return noisy.masked_fill(~mask, 0.0)


def lwm_tokens(sparse):
    values = torch.view_as_real(sparse).float()
    scale = values.abs().flatten(1).amax(1).clamp_min(torch.finfo(torch.float32).eps)
    values = values / scale[:, None, None, None, None]
    batch = len(values)
    patches = values.reshape(batch, 4, 4, 18, 4, 2)
    patches = patches.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(batch, 72, 32)
    cls = torch.full((batch, 1, 32), 0.2, device=values.device)
    return torch.cat((cls, patches), 1)


def contrawimae_matrix(sparse):
    matrix = sparse.reshape(len(sparse), 16, 72)
    components = torch.view_as_real(matrix)
    scale = components.abs().flatten(1).amax(1).clamp_min(torch.finfo(components.dtype).eps)
    return matrix / scale[:, None, None]


class BeamHead(nn.Module):
    def __init__(self, input_dimension):
        super().__init__()
        self.input_proj = nn.Linear(input_dimension, 256)
        self.position = PositionalEncoding(256)
        layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=8, dim_feedforward=512, dropout=0.1,
            activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.norm = nn.LayerNorm(256)
        self.classifier = nn.Linear(256, NUM_CLASSES)

    def forward(self, features):
        hidden = self.encoder(self.position(self.input_proj(features)))
        return self.classifier(self.norm(hidden.mean(1)))


class LwmPipeline(nn.Module):
    def __init__(self, use_rafc):
        super().__init__()
        self.backbone = LWM(
            element_length=32, d_model=128, n_layers=12,
            max_len=73, n_heads=8, dropout=0.1,
        )
        self.router = DynamicFeatureFusion(128) if use_rafc else None
        self.head = BeamHead(128)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, sparse, return_alpha=False):
        output = self.backbone.embedding(lwm_tokens(sparse))
        captured = {}
        for index, layer in enumerate(self.backbone.layers):
            output, _ = layer(output)
            if index in LAYER_IDS:
                captured[f"layer_{index}"] = output[:, 1:]
        features = output[:, 1:]
        alpha = None
        if self.router is not None:
            features, alpha = self.router(captured)
        logits = self.head(features)
        return (logits, alpha) if return_alpha else logits


class ContraPipeline(nn.Module):
    def __init__(self, use_rafc):
        super().__init__()
        self.backbone = ContraWiMAE(
            patch_size=(16, 1), encoder_dim=64, encoder_layers=12,
            encoder_nhead=16, decoder_layers=4, decoder_nhead=8,
            mask_ratio=0.9, contrastive_dim=64, temperature=0.2,
            snr_min=5.0, snr_max=40.0, device=torch.device("cpu"), max_len=144,
        )
        self.router = DynamicFeatureFusion(64) if use_rafc else None
        self.head = BeamHead(64)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, sparse, return_alpha=False):
        matrix = contrawimae_matrix(sparse)
        output = self.backbone.encoder.linear_1(self.backbone.patcher(matrix))
        output = self.backbone.encoder.positional_encoding(output)
        captured = {}
        for index, layer in enumerate(self.backbone.encoder.transformer.layers):
            output = layer(output)
            if index in LAYER_IDS:
                captured[f"layer_{index}"] = output
        norm = self.backbone.encoder.transformer.norm
        if norm is not None:
            output = norm(output)
        alpha = None
        if self.router is not None:
            output, alpha = self.router(captured)
        logits = self.head(output)
        return (logits, alpha) if return_alpha else logits


def metrics(confusion):
    matrix = confusion.double()
    true_positive = matrix.diag()
    support = matrix.sum(1)
    predicted = matrix.sum(0)
    precision = true_positive / predicted.clamp_min(1.0)
    recall = true_positive / support.clamp_min(1.0)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-12)
    active = support > 0
    return {
        "top1_accuracy": true_positive.sum().div(matrix.sum()).item(),
        "macro_f1": f1[active].mean().item(),
        "weighted_f1": (f1 * support).sum().div(support.sum()).item(),
    }


@torch.no_grad()
def evaluate(model, loader, device, snr, seed):
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed + int(snr) * 1000)
    confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    loss_sum = 0.0
    count = 0
    for clean, label in loader:
        clean = clean.to(device)
        label = label.to(device)
        logits = model(sparse_observation(clean, snr, generator))
        loss_sum += nn.functional.cross_entropy(logits, label, reduction="sum").item()
        prediction = logits.argmax(1)
        bins = torch.bincount(
            (label * NUM_CLASSES + prediction).cpu(), minlength=NUM_CLASSES ** 2
        )
        confusion += bins.reshape(NUM_CLASSES, NUM_CLASSES)
        count += len(label)
    result = metrics(confusion)
    result["cross_entropy"] = loss_sum / count
    return result


def make_model(backbone, use_rafc):
    return LwmPipeline(use_rafc) if backbone == "lwm1_1" else ContraPipeline(use_rafc)


def load_backbone(model, backbone, checkpoint):
    if not checkpoint.exists():
        raise FileNotFoundError(f"Backbone checkpoint not found: {checkpoint}")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=(backbone == "lwm1_1"))
    if backbone == "lwm1_1":
        state = saved.get("model", saved)
    else:
        state = saved.get("model_state_dict", saved.get("model", saved))
    model.backbone.load_state_dict(state, strict=True)
    for parameter in model.backbone.parameters():
        parameter.requires_grad_(False)
    model.backbone.eval()


def loader(dataset, indices, batch_size):
    return DataLoader(
        Subset(dataset, indices.astype(np.int64)), batch_size=batch_size,
        shuffle=False, num_workers=0, pin_memory=True,
    )


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def train_model(args, model, dataset, split, device, validation, test):
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=args.learning_rate, weight_decay=args.weight_decay
    )
    output_dir = args.output_dir or (
        PROJECT_ROOT / "runs"
        / f"beam_prediction_{args.backbone}_{args.variant}_seed{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path, last_path = output_dir / "best.pt", output_dir / "last.pt"
    result_path = output_dir / "results.json"
    best, best_epoch, bad, history, start_epoch = -math.inf, 0, 0, [], 1
    if args.resume and last_path.exists():
        saved = torch.load(last_path, map_location="cpu", weights_only=True)
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        best, best_epoch, bad = saved["best"], saved["best_epoch"], saved["bad"]
        history, start_epoch = saved["history"], saved["epoch"] + 1

    print(json.dumps({
        "task": "beam_prediction", "backbone": args.backbone,
        "variant": args.variant, "seed": args.seed, "num_beams": NUM_CLASSES,
        "backbone_frozen": True, "rafc_layers": list(LAYER_IDS) if args.variant == "rafc" else None,
        "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        "train": len(split["train"]), "validation": len(split["validation"]),
        "test": len(split["test"]), "train_snr_db": "uniform(0,30)",
        "validation_snrs_db": args.snrs,
    }, indent=2), flush=True)

    if args.smoke_test:
        smoke = DataLoader(Subset(dataset, split["train"][:2]), batch_size=2)
        clean, labels = next(iter(smoke))
        clean, labels = clean.to(device), labels.to(device)
        generator = torch.Generator(device=device).manual_seed(args.seed)
        logits = model(sparse_observation(clean, 15.0, generator))
        loss = nn.functional.cross_entropy(logits, labels)
        loss.backward()
        print({"logits": tuple(logits.shape), "loss": loss.item()}, flush=True)
        return None

    for epoch in range(start_epoch, args.epochs + 1):
        training = DataLoader(
            Subset(dataset, split["train"].astype(np.int64)),
            batch_size=args.train_batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(args.seed + epoch),
            num_workers=0, pin_memory=True,
        )
        generator = torch.Generator(device=device).manual_seed(50000 + args.seed + epoch)
        model.train()
        confusion = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
        loss_sum, count = 0.0, 0
        alpha_sum = torch.zeros(3, device=device) if args.variant == "rafc" else None
        alpha_count = 0
        for clean, labels in training:
            clean, labels = clean.to(device), labels.to(device)
            snr = torch.rand(len(clean), device=device, generator=generator) * 30.0
            observation = sparse_observation(clean, snr, generator)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                if args.variant == "rafc":
                    logits, alpha = model(observation, return_alpha=True)
                    alpha_sum += alpha.detach().float().sum(0)
                    alpha_count += len(alpha)
                else:
                    logits = model(observation)
                loss = nn.functional.cross_entropy(logits.float(), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            predictions = logits.detach().argmax(1)
            confusion += torch.bincount(
                (labels * NUM_CLASSES + predictions).cpu(),
                minlength=NUM_CLASSES ** 2,
            ).reshape(NUM_CLASSES, NUM_CLASSES)
            loss_sum += loss.item() * len(labels)
            count += len(labels)

        validation_by_snr = {
            str(snr): evaluate(model, validation, device, snr, 10000 + args.seed)
            for snr in args.snrs
        }
        mean_macro_f1 = float(np.mean([
            value["macro_f1"] for value in validation_by_snr.values()
        ]))
        improved = mean_macro_f1 > best + 1e-8
        if improved:
            best, best_epoch, bad = mean_macro_f1, epoch, 0
            torch.save({
                "model": model.state_dict(), "epoch": epoch,
                "best_val_mean_macro_f1": best,
            }, best_path)
        else:
            bad += 1
        row = {
            "epoch": epoch, "train_cross_entropy": loss_sum / count,
            **{f"train_{key}": value for key, value in metrics(confusion).items()},
            "val_mean_macro_f1": mean_macro_f1,
            "validation_by_snr": validation_by_snr, "bad_epochs": bad,
        }
        if alpha_sum is not None:
            row["alpha"] = (alpha_sum / alpha_count).cpu().tolist()
        history.append(row)
        torch.save({
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "epoch": epoch, "best": best, "best_epoch": best_epoch,
            "bad": bad, "history": history,
        }, last_path)
        atomic_json(result_path, {
            "status": "training", "best_epoch": best_epoch,
            "best_val_mean_macro_f1": best, "history": history,
        })
        print(
            f"[beam/{args.backbone}/{args.variant}] E{epoch:04d}/{args.epochs} "
            f"ce={loss_sum / count:.6f} val_mean_macro_f1={mean_macro_f1:.6f} "
            f"best={best:.6f} bad={bad}/{args.patience}"
            + (" saved" if improved else ""), flush=True,
        )
        if bad >= args.patience:
            break

    saved = torch.load(best_path, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["model"], strict=True)
    model.to(device)
    output = {
        "task": "beam_prediction", "codebook_size": NUM_CLASSES,
        "backbone": args.backbone, "variant": args.variant,
        "checkpoint": str(best_path), "checkpoint_epoch": saved.get("epoch"),
        "test_by_snr": {
            str(snr): evaluate(model, test, device, snr, 20000 + args.seed)
            for snr in args.snrs
        },
    }
    atomic_json(result_path, output)
    return output


def main():
    parser = argparse.ArgumentParser(
        description="Train or evaluate released 64-codeword beam predictors."
    )
    parser.add_argument("--backbone", choices=("lwm1_1", "contrawimae"), required=True)
    parser.add_argument("--variant", choices=("baseline", "rafc"), required=True)
    parser.add_argument("--checkpoint", type=Path, help="Task checkpoint for evaluation.")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--snrs", type=int, nargs="+", default=[0, 5, 10, 15, 20, 25, 30])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lwm-checkpoint", type=Path,
                        default=PROJECT_ROOT / "weights/pretrained/lwm1_1.pth")
    parser.add_argument("--contra-checkpoint", type=Path,
                        default=PROJECT_ROOT / "weights/pretrained/contrawimae.pt")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    manifest_path = args.manifest if args.manifest.is_absolute() else args.data_root / args.manifest
    manifest = json.loads(manifest_path.read_text())
    if int(manifest.get("num_beams", NUM_CLASSES)) != NUM_CLASSES:
        raise ValueError("This script requires a 64-codeword manifest")
    dataset = BeamDataset(args.data_root, manifest["records"])
    split_file = np.load(args.data_root / manifest["split"]["path"])
    split = {key: split_file[key].astype(np.int64)
             for key in ("train", "validation", "test")}
    validation = loader(dataset, split["validation"], args.batch_size)
    test = loader(dataset, split["test"], args.batch_size)
    device = torch.device(args.device)
    model = make_model(args.backbone, args.variant == "rafc")

    if args.train:
        pretrained = (args.lwm_checkpoint if args.backbone == "lwm1_1"
                      else args.contra_checkpoint)
        load_backbone(model, args.backbone, pretrained)
        model.to(device)
        output = train_model(args, model, dataset, split, device, validation, test)
        if output is not None:
            print(json.dumps(output, indent=2))
        return

    if args.checkpoint is None:
        raise SystemExit("--checkpoint is required unless --train is used")
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(saved["model"], strict=True)
    model.to(device)
    output = {
        "task": "beam_prediction", "codebook_size": NUM_CLASSES,
        "backbone": args.backbone, "variant": args.variant,
        "checkpoint": str(args.checkpoint), "checkpoint_epoch": saved.get("epoch"),
        "test_by_snr": {
            str(snr): evaluate(model, test, device, snr, 20000 + args.seed)
            for snr in args.snrs
        },
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
