#!/usr/bin/env python3
"""LWMv1.1/ContraWiMAE channel estimation and 14-to-14 prediction.

Each OFDM symbol is independently encoded by a frozen spatial-frequency
foundation model.  The 14 aligned snapshot features are concatenated along
the feature dimension and passed to a two-layer Transformer task head.
"""

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
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.contrawimae import ContraWiMAE
from models.lwm1_1 import LWM


SPLIT_SEED = 2026
TIME = 14
PILOT_SYMBOLS = (0, 11)
PILOT_COMB = 4
TEST_SNRS = (0, 5, 10, 15, 20, 25, 30)
LAYER_IDS = (0, 5, 11)
LAYER_NAMES = ("layer_0", "layer_5", "layer_11")


class PositionalEncoding(nn.Module):
    def __init__(self, dimension: int, max_length: int = 1024):
        super().__init__()
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, dimension, 2, dtype=torch.float32)
            * (-math.log(10000.0) / dimension)
        )
        encoding = torch.zeros(max_length, dimension)
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("pe", encoding.unsqueeze(0))

    def forward(self, values):
        return values + self.pe[:, : values.shape[1]]


class ChannelHead(nn.Module):
    def __init__(self, in_dim, out_dim, token_count):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, 256)
        self.pos_encoding = PositionalEncoding(256)
        layer = nn.TransformerEncoderLayer(
            256, 8, 512, dropout=0.1, activation="gelu", batch_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.output_proj = nn.Linear(256, out_dim)
        self.seq_len = token_count

    def forward(self, values):
        values = self.pos_encoding(self.input_proj(values))
        return self.output_proj(self.encoder(values))


class FeatureFusion(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3 * dimension, max(dimension // 4, 1)),
            nn.ReLU(inplace=True),
            nn.Linear(max(dimension // 4, 1), 3),
        )

    def forward(self, features):
        values = list(features.values())
        alpha = self.mlp(torch.cat([value.mean(1) for value in values], -1)).softmax(-1)
        fused = sum(alpha[:, index, None, None] * value for index, value in enumerate(values))
        return fused, alpha


class EstimationDataset(Dataset):
    def __init__(self, path: Path):
        self.array = np.load(path, mmap_mode="r")

    def __len__(self):
        return len(self.array)

    def __getitem__(self, index):
        return torch.from_numpy(np.array(self.array[index], copy=True))


class PredictionDataset(Dataset):
    def __init__(self, records: list[dict], data_root: Path):
        paths = [Path(record["path"]) for record in records]
        self.arrays = [
            np.load(path if path.is_absolute() else data_root / path, mmap_mode="r")
            for path in paths
        ]
        self.ends = np.cumsum([len(array) for array in self.arrays]).tolist()

    def __len__(self):
        return self.ends[-1]

    def __getitem__(self, index):
        part = bisect_right(self.ends, index)
        local = index - (self.ends[part - 1] if part else 0)
        full = torch.from_numpy(np.array(self.arrays[part][local], copy=True))
        return full[..., :TIME], full[..., TIME: 2 * TIME]


def pilot_mask(x: torch.Tensor) -> torch.Tensor:
    mask = torch.zeros((1, 1, x.shape[2], x.shape[3], x.shape[4]),
                       dtype=torch.bool, device=x.device)
    for tx in range(x.shape[2]):
        for symbol in PILOT_SYMBOLS:
            mask[0, 0, tx, tx::PILOT_COMB, symbol] = True
    return mask.expand_as(x)


def sparse_observation(clean: torch.Tensor, snr_db, generator) -> torch.Tensor:
    mask = pilot_mask(clean)
    mf = mask.float()
    dims = tuple(range(1, clean.ndim))
    count = mf.sum(dims, keepdim=True).clamp_min(1.0)
    signal_power = (clean.abs().square() * mf).sum(dims, keepdim=True) / count
    snr = torch.as_tensor(snr_db, device=clean.device, dtype=clean.real.dtype)
    if snr.ndim == 1:
        snr = snr.reshape(clean.shape[0], *([1] * (clean.ndim - 1)))
    noise_power = signal_power / torch.pow(10.0, snr / 10.0)
    real = torch.randn(clean.shape, device=clean.device, generator=generator)
    imag = torch.randn(clean.shape, device=clean.device, generator=generator)
    noisy = clean + torch.complex(real, imag) * torch.sqrt(noise_power / 2.0)
    return noisy.masked_fill(~mask, 0.0)


def nmse_real(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(1, prediction.ndim))
    return ((prediction - target).square().sum(dims) /
            target.square().sum(dims).clamp_min(1e-10))


def nmse_to_db(value):
    return 10.0 * math.log10(max(float(value), 1e-30))


def component_max_scale(x: torch.Tensor) -> torch.Tensor:
    values = torch.view_as_real(x).float()
    return values.abs().flatten(1).amax(1).clamp_min(torch.finfo(torch.float32).eps)


def lwm_raw_patches(snapshot: torch.Tensor) -> torch.Tensor:
    """[B,4,4,F] complex -> [B,4*(F/4),32], LWM pretraining order."""
    batch, rx, tx, freq = snapshot.shape
    if (rx, tx) != (4, 4) or freq % 4:
        raise ValueError(f"Unexpected LWM snapshot shape {tuple(snapshot.shape)}")
    values = torch.view_as_real(snapshot).float()
    blocks = values.reshape(batch, 4, 4, freq // 4, 4, 2)
    return blocks.permute(0, 1, 3, 2, 4, 5).contiguous().reshape(batch, 4 * (freq // 4), 32)


class LWMBackbone(nn.Module):
    feature_dim = 128
    patch_dim = 32

    def __init__(self, checkpoint: Path):
        super().__init__()
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model = LWM(element_length=32, d_model=128, n_layers=12,
                         max_len=73, n_heads=8, dropout=0.1)
        self.model.load_state_dict(saved["model"] if "model" in saved else saved, strict=True)
        self.checkpoint = str(checkpoint)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()
        return self

    def _position(self, n_frequency_patches: int, device, dtype):
        weight = self.model.embedding.pos_embed.weight
        cls = weight[:1]
        grid = weight[1:73].reshape(4, 18, 128)
        if n_frequency_patches != 18:
            grid = F.interpolate(
                grid.permute(2, 0, 1).unsqueeze(0),
                size=(4, n_frequency_patches), mode="bicubic", align_corners=False,
            ).squeeze(0).permute(1, 2, 0)
        return torch.cat((cls, grid.reshape(-1, 128)), dim=0).to(device=device, dtype=dtype)

    @torch.no_grad()
    def encode(self, normalized_snapshot: torch.Tensor):
        patches = lwm_raw_patches(normalized_snapshot)
        cls = torch.full((len(patches), 1, 32), 0.2,
                         device=patches.device, dtype=patches.dtype)
        tokens = torch.cat((cls, patches), dim=1)
        output = self.model.embedding.proj(tokens.float())
        output = self.model.embedding.norm(
            output + self._position(normalized_snapshot.shape[-1] // 4,
                                    output.device, output.dtype)[None]
        )
        captured = {}
        for index, layer in enumerate(self.model.layers):
            output, _ = layer(output)
            if index in LAYER_IDS:
                captured[f"layer_{index}"] = output[:, 1:].detach()
        return output[:, 1:].detach(), captured

    def raw_target(self, snapshot: torch.Tensor):
        return lwm_raw_patches(snapshot)


class ContraBackbone(nn.Module):
    feature_dim = 64
    patch_dim = 16

    def __init__(self, checkpoint: Path):
        super().__init__()
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.model = ContraWiMAE(
            patch_size=(16, 1), encoder_dim=64, encoder_layers=12,
            encoder_nhead=16, decoder_layers=4, decoder_nhead=8,
            mask_ratio=0.9, contrastive_dim=64, temperature=0.2,
            snr_min=5.0, snr_max=40.0, device=torch.device("cpu"), max_len=144,
        )
        state = saved.get("model_state_dict", saved.get("model", saved))
        self.model.load_state_dict(state, strict=True)
        self.checkpoint = str(checkpoint)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.eval()

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()
        return self

    def _position(self, freq: int, device, dtype):
        base = self.model.encoder.positional_encoding.position_embeddings
        if freq == 72:
            return base.to(device=device, dtype=dtype)
        # Pretraining order is [real frequencies, imaginary frequencies].
        components = base.reshape(1, 2, 72, 64).permute(0, 1, 3, 2)
        components = F.interpolate(
            components.reshape(2, 64, 72), size=freq,
            mode="linear", align_corners=False,
        ).reshape(1, 2, 64, freq).permute(0, 1, 3, 2)
        return components.reshape(1, 2 * freq, 64).to(device=device, dtype=dtype)

    def _patches(self, snapshot: torch.Tensor):
        matrix = snapshot.reshape(len(snapshot), 16, snapshot.shape[-1])
        return self.model.patcher(matrix).float()

    @torch.no_grad()
    def encode(self, normalized_snapshot: torch.Tensor):
        patches = self._patches(normalized_snapshot)
        output = self.model.encoder.linear_1(patches)
        output = output + self._position(normalized_snapshot.shape[-1], output.device, output.dtype)
        captured = {}
        for index, layer in enumerate(self.model.encoder.transformer.layers):
            output = layer(output)
            if index in LAYER_IDS:
                captured[f"layer_{index}"] = output.detach()
        norm = self.model.encoder.transformer.norm
        if norm is not None:
            output = norm(output)
        return output.detach(), captured

    def raw_target(self, snapshot: torch.Tensor):
        return self._patches(snapshot)


class SnapshotPipeline(nn.Module):
    def __init__(self, mode, freq, encoder_chunk, lwm_checkpoint, contra_checkpoint):
        super().__init__()
        self.mode = mode
        self.freq = freq
        self.encoder_chunk = encoder_chunk
        if mode.startswith("lwm"):
            self.backbone = LWMBackbone(lwm_checkpoint)
        else:
            self.backbone = ContraBackbone(contra_checkpoint)
        self.use_rafc = mode.endswith("_rafc")
        self.router = (FeatureFusion(self.backbone.feature_dim)
            if self.use_rafc else None)
        token_count = 4 * (freq // 4) if mode.startswith("lwm") else 2 * freq
        self.token_count = token_count
        self.head = ChannelHead(
            TIME * self.backbone.feature_dim,
            TIME * self.backbone.patch_dim,
            token_count,
        )

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def _features(self, observation: torch.Tensor):
        batch = len(observation)
        scale = component_max_scale(observation)
        normalized = observation / scale[:, None, None, None, None]
        snapshots = normalized.permute(0, 4, 1, 2, 3).reshape(
            batch * TIME, 4, 4, self.freq)
        outputs, alphas = [], []
        for start in range(0, len(snapshots), self.encoder_chunk):
            last, layers = self.backbone.encode(snapshots[start:start + self.encoder_chunk])
            if self.router is not None:
                fused, alpha = self.router(layers)
                outputs.append(fused)
                alphas.append(alpha)
            else:
                outputs.append(last)
        encoded = torch.cat(outputs, dim=0).reshape(
            batch, TIME, self.token_count, self.backbone.feature_dim)
        aligned = encoded.permute(0, 2, 1, 3).reshape(
            batch, self.token_count, TIME * self.backbone.feature_dim)
        alpha = torch.cat(alphas, dim=0) if alphas else None
        return aligned, scale, alpha

    def forward(self, observation: torch.Tensor, return_alpha=False):
        features, scale, alpha = self._features(observation)
        prediction = self.head(features).reshape(
            len(observation), self.token_count, TIME, self.backbone.patch_dim)
        prediction = prediction * scale[:, None, None, None]
        return (prediction, alpha) if return_alpha else prediction

    def target(self, clean: torch.Tensor):
        batch = len(clean)
        snapshots = clean.permute(0, 4, 1, 2, 3).reshape(batch * TIME, 4, 4, self.freq)
        chunks = []
        for start in range(0, len(snapshots), self.encoder_chunk):
            chunks.append(self.backbone.raw_target(snapshots[start:start + self.encoder_chunk]))
        patches = torch.cat(chunks, dim=0).reshape(
            batch, TIME, self.token_count, self.backbone.patch_dim)
        return patches.permute(0, 2, 1, 3).contiguous()

    def trainable_state(self):
        return {
            "head": self.head.state_dict(),
            "router": None if self.router is None else self.router.state_dict(),
        }

    def load_trainable_state(self, state):
        self.head.load_state_dict(state["head"])
        if self.router is not None:
            self.router.load_state_dict(state["router"])


def resolve_data_path(data_root: Path, manifest_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    candidates = (data_root / path, manifest_path.parent / path)
    return next((candidate for candidate in candidates if candidate.exists()), candidates[0])


def make_data(task: str, data_root: Path, manifest_path: Path):
    manifest = json.loads(manifest_path.read_text())
    if task == "estimation":
        dataset = EstimationDataset(resolve_data_path(data_root, manifest_path, manifest["path"]))
        freq = 72
    else:
        records = [
            {**record, "path": str(resolve_data_path(data_root, manifest_path, record["path"]))}
            for record in manifest["records"]
        ]
        dataset = PredictionDataset(records, data_root)
        freq = 144
    indices = np.random.default_rng(SPLIT_SEED).permutation(len(dataset))
    n_train, n_val = int(0.70 * len(indices)), int(0.15 * len(indices))
    split = {
        "train": indices[:n_train],
        "validation": indices[n_train:n_train + n_val],
        "test": indices[n_train + n_val:],
    }
    return dataset, split, freq, manifest


def batch_clean(task: str, batch, device):
    if task == "estimation":
        clean = batch.to(device, non_blocking=True)
        return clean, clean
    history, future = batch
    return history.to(device, non_blocking=True), future.to(device, non_blocking=True)


@torch.no_grad()
def evaluate(model, loader, task, device, snr, seed):
    model.eval()
    generator = torch.Generator(device=device).manual_seed(seed + int(snr) * 1000)
    total = 0.0
    count = 0
    for batch in loader:
        source, target = batch_clean(task, batch, device)
        observation = sparse_observation(source, float(snr), generator)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction = model(observation)
            clean_tokens = model.target(target)
            values = nmse_real(prediction.float(), clean_tokens.float())
        total += values.sum().item()
        count += len(values)
    return total / count


def atomic_json(path: Path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(
        description="Train or evaluate LWMv1.1 and ContraWiMAE channel tasks."
    )
    parser.add_argument("--task", choices=("estimation", "prediction"), required=True)
    parser.add_argument("--mode", choices=("lwm", "lwm_rafc", "contrawimae", "contrawimae_rafc"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--encoder-chunk", type=int, default=256)
    parser.add_argument("--run-name", default="snapshot_foundation_ce_cp_1000ep_p30")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--lwm-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "weights/pretrained/lwm1_1.pth",
    )
    parser.add_argument(
        "--contra-checkpoint",
        type=Path,
        default=PROJECT_ROOT / "weights/pretrained/contrawimae.pt",
    )
    parser.add_argument("--checkpoint", type=Path, help="Task checkpoint for --eval-only.")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device(args.device)
    manifest_path = args.manifest
    if not manifest_path.is_absolute():
        manifest_path = args.data_root / manifest_path
    dataset, split, freq, manifest = make_data(args.task, args.data_root, manifest_path)
    if args.validate_only:
        print(json.dumps({
            "status": "valid", "task": args.task, "frequency": freq,
            "samples": len(dataset), "split": {key: len(value) for key, value in split.items()},
        }, indent=2))
        return
    required_backbone = args.lwm_checkpoint if args.mode.startswith("lwm") else args.contra_checkpoint
    if not required_backbone.exists():
        raise FileNotFoundError(f"Backbone checkpoint not found: {required_backbone}")
    model = SnapshotPipeline(
        args.mode, freq, args.encoder_chunk, args.lwm_checkpoint, args.contra_checkpoint
    ).to(device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-4, weight_decay=1e-4)

    run_root = PROJECT_ROOT / "runs" / args.run_name
    output = ((run_root / args.task / args.mode) if args.seed == 2026 else
              (run_root / f"seed_{args.seed}" / args.task / args.mode))
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "split_indices.npz", **split)
    result_path = output / "results.json"
    best_path = output / "best.pt"
    last_path = output / "last.pt"

    print(json.dumps({
        "task": args.task, "mode": args.mode, "device": args.device,
        "seed": args.seed, "split_seed": SPLIT_SEED,
        "frequency": freq, "symbols_in": TIME, "symbols_out": TIME,
        "token_count_per_symbol": model.token_count,
        "feature_concat": f"14x{model.backbone.feature_dim}",
        "head": "2-layer Transformer, d_model=256, heads=8, d_ff=512",
        "backbone_checkpoint": model.backbone.checkpoint,
        "backbone_frozen": True, "rafc_layers": list(LAYER_NAMES) if model.use_rafc else None,
        "trainable_parameters": sum(p.numel() for p in trainable),
        "train": len(split["train"]), "validation": len(split["validation"]),
        "test": len(split["test"]), "pilot_symbols": list(PILOT_SYMBOLS),
        "pilot_comb": PILOT_COMB,
    }, indent=2), flush=True)

    if args.smoke_test:
        loader = DataLoader(Subset(dataset, split["train"][:2]), batch_size=2)
        source, target = batch_clean(args.task, next(iter(loader)), device)
        generator = torch.Generator(device=device).manual_seed(args.seed)
        observation = sparse_observation(source, 15.0, generator)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            prediction, alpha = model(observation, return_alpha=True)
            clean_tokens = model.target(target)
            loss = nmse_real(prediction.float(), clean_tokens.float()).mean()
        loss.backward()
        print({"prediction": tuple(prediction.shape), "target": tuple(clean_tokens.shape),
               "nmse_db": nmse_to_db(loss.item()),
               "alpha": None if alpha is None else tuple(alpha.shape)}, flush=True)
        return

    validation = DataLoader(Subset(dataset, split["validation"]), args.eval_batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)
    test = DataLoader(Subset(dataset, split["test"]), args.eval_batch_size,
                      shuffle=False, num_workers=0, pin_memory=True)
    if args.eval_only:
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required with --eval-only")
        saved = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_trainable_state(saved["trainable"])
        values = {
            str(snr): nmse_to_db(evaluate(model, test, args.task, device, snr, 20000 + args.seed))
            for snr in TEST_SNRS
        }
        print(json.dumps({
            "task": args.task, "mode": args.mode, "checkpoint": str(args.checkpoint),
            "test_nmse_db": values,
        }, indent=2))
        return
    best, best_epoch, bad, history, start_epoch = math.inf, 0, 0, [], 1
    if args.resume and last_path.exists():
        saved = torch.load(last_path, map_location=device, weights_only=True)
        model.load_trainable_state(saved["trainable"])
        optimizer.load_state_dict(saved["optimizer"])
        best, best_epoch, bad = saved["best"], saved["best_epoch"], saved["bad"]
        history, start_epoch = saved["history"], saved["epoch"] + 1

    for epoch in range(start_epoch, args.epochs + 1):
        train = DataLoader(
            Subset(dataset, split["train"]), args.train_batch_size, shuffle=True,
            generator=torch.Generator().manual_seed(args.seed + epoch),
            num_workers=0, pin_memory=True,
        )
        generator = torch.Generator(device=device).manual_seed(50000 + args.seed + epoch)
        model.train()
        total, count = 0.0, 0
        alpha_sum = torch.zeros(3, device=device) if model.use_rafc else None
        alpha_count = 0
        for batch in train:
            source, target = batch_clean(args.task, batch, device)
            snr = torch.rand(len(source), device=device, generator=generator) * 30.0
            observation = sparse_observation(source, snr, generator)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                if model.use_rafc:
                    prediction, alpha = model(observation, return_alpha=True)
                    alpha_sum += alpha.detach().float().sum(0)
                    alpha_count += len(alpha)
                else:
                    prediction = model(observation)
                clean_tokens = model.target(target)
                loss = nmse_real(prediction.float(), clean_tokens.float()).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            total += loss.item() * len(source)
            count += len(source)

        train_nmse = total / count
        val_nmse = evaluate(model, validation, args.task, device, 15, 10000 + args.seed)
        improved = val_nmse < best
        if improved:
            best, best_epoch, bad = val_nmse, epoch, 0
            temporary = best_path.with_suffix(".pt.tmp")
            torch.save({"trainable": model.trainable_state(), "epoch": epoch, "best": best}, temporary)
            temporary.replace(best_path)
        else:
            bad += 1
        row = {"epoch": epoch, "train_nmse_db": nmse_to_db(train_nmse),
               "val_nmse_db_at_15db": nmse_to_db(val_nmse),
               "best_val_nmse_db_at_15db": nmse_to_db(best), "bad_epochs": bad}
        if alpha_sum is not None:
            row["alpha"] = (alpha_sum / alpha_count).cpu().tolist()
        history.append(row)
        torch.save({"trainable": model.trainable_state(), "optimizer": optimizer.state_dict(),
                    "epoch": epoch, "best": best, "best_epoch": best_epoch,
                    "bad": bad, "history": history}, last_path)
        atomic_json(result_path, {"status": "training", "task": args.task,
                                  "mode": args.mode, "history": history})
        print(f"[{args.task}/{args.mode}] E{epoch:04d}/{args.epochs} "
              f"train={nmse_to_db(train_nmse):.4f}dB val15={nmse_to_db(val_nmse):.4f}dB "
              f"best={nmse_to_db(best):.4f}dB bad={bad}/{args.patience}" +
              (" saved" if improved else "") +
              (f" alpha={row['alpha']}" if "alpha" in row else ""), flush=True)
        if bad >= args.patience:
            print(f"[{args.task}/{args.mode}] EARLY_STOP epoch={epoch} best_epoch={best_epoch}", flush=True)
            break

    saved = torch.load(best_path, map_location=device, weights_only=True)
    model.load_trainable_state(saved["trainable"])
    test_nmse = {str(snr): evaluate(model, test, args.task, device, snr, 20000 + args.seed)
                 for snr in TEST_SNRS}
    alpha = None
    if model.use_rafc:
        model.eval()
        alpha_sum = torch.zeros(3, device=device)
        alpha_count = 0
        generator = torch.Generator(device=device).manual_seed(35000 + args.seed)
        with torch.no_grad():
            for batch in test:
                source, _ = batch_clean(args.task, batch, device)
                observation = sparse_observation(source, 15.0, generator)
                _, value = model(observation, return_alpha=True)
                alpha_sum += value.float().sum(0)
                alpha_count += len(value)
        alpha = (alpha_sum / alpha_count).cpu().tolist()
    payload = {
        "status": "complete", "task": args.task, "mode": args.mode,
        "seed": args.seed, "split_seed": SPLIT_SEED,
        "frequency": freq, "symbols_in": TIME, "symbols_out": TIME,
        "snapshot_encoding": "each OFDM symbol independently through frozen backbone",
        "feature_aggregation": "14 aligned snapshot features concatenated along feature dimension",
        "head": {"type": "TransformerEncoder", "layers": 2, "d_model": 256,
                 "heads": 8, "d_ff": 512},
        "backbone": type(model.backbone).__name__,
        "backbone_checkpoint": model.backbone.checkpoint,
        "backbone_frozen": True, "rafc_layers": list(LAYER_NAMES) if model.use_rafc else None,
        "pilot_symbols": list(PILOT_SYMBOLS), "pilot_comb": PILOT_COMB,
        "train_snr_db": "Uniform[0,30]", "validation_snr_db": 15,
        "test_snrs_db": list(TEST_SNRS), "loss_metric": "physical-domain NMSE (reported in dB)",
        "max_epochs": args.epochs, "patience": args.patience,
        "best_epoch": saved["epoch"],
        "best_val_nmse_db_at_15db": nmse_to_db(saved["best"]),
        "test_nmse_db": {snr: nmse_to_db(value) for snr, value in test_nmse.items()},
        "rafc_alpha_15db": alpha,
        "trainable_parameters": sum(p.numel() for p in trainable),
        "split": {key: len(value) for key, value in split.items()},
        "history": history,
    }
    atomic_json(result_path, payload)
    print(f"[{args.task}/{args.mode}] COMPLETE best_epoch={saved['epoch']} "
          f"TEST_DB={payload['test_nmse_db']} alpha={alpha}", flush=True)


if __name__ == "__main__":
    main()
