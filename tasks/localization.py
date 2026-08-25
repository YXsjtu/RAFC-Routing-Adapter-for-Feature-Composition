"""Frozen LWM1.1/ContraWiMAE localization on City16 with a Transformer head."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import RAFC, normalize_max_component, root_mean_square
from models.contrawimae import ContraWiMAE
from models.lwm1_1 import LWM

LAYER_IDS = (0, 5, 11)
NUM_RX = 2
NUM_TX = 32
NUM_SUBCARRIERS = 144
NUM_SYMBOLS = 14
LWM_TOKEN_COUNT = NUM_RX * (NUM_TX // 16) * NUM_SUBCARRIERS
CONTRA_TOKEN_COUNT = 2 * (NUM_RX * NUM_TX // 16) * NUM_SUBCARRIERS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--mode",
        choices=("lwm11", "lwm11_rafc", "contrawimae", "contrawimae_rafc"),
        required=True,
    )
    root = Path(__file__).resolve().parents[1]
    p.add_argument("--output", type=Path, default=root / "runs/localization")
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--generate-cache", action="store_true")
    p.add_argument("--data-source-root", type=Path)
    p.add_argument("--samples-per-bs", type=int, default=15000)
    p.add_argument(
        "--lwm-checkpoint", type=Path,
        default=root / "weights/pretrained/lwm1_1.pth",
    )
    p.add_argument(
        "--contra-checkpoint", type=Path,
        default=root / "weights/pretrained/contrawimae.pt",
    )
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=20260817)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--resume", type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--eval-only", action="store_true")
    return p.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def split_indices(size, seed):
    order = torch.randperm(size, generator=torch.Generator().manual_seed(seed))
    train_size, validation_size = int(0.70 * size), int(0.15 * size)
    return (
        order[:train_size],
        order[train_size : train_size + validation_size],
        order[train_size + validation_size :],
    )


def metrics(errors):
    return {
        "mean_m": float(errors.mean()),
        "median_m": float(np.median(errors)),
        "p80_m": float(np.percentile(errors, 80)),
        "p90_m": float(np.percentile(errors, 90)),
        "rmse_m": float(np.sqrt(np.mean(errors**2))),
    }


class ScaleTransformerLocalizationHead(nn.Module):
    def __init__(self, in_dim, d_model=256, max_len=160, dropout=0.1):
        super().__init__()
        self.input_projection = nn.Linear(in_dim, d_model)
        self.loc_query = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.normal_(self.loc_query, std=0.02)
        self.position = nn.Parameter(torch.zeros(1, max_len + 1, d_model))
        self.scale_encoder = nn.Sequential(
            nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        nn.init.zeros_(self.scale_encoder[-1].weight)
        nn.init.zeros_(self.scale_encoder[-1].bias)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=1024,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=2, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_model)
        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Linear(128, 3),
            nn.Sigmoid(),
        )

    def forward(self, features, scale):
        x = self.input_projection(features)
        query = self.loc_query.expand(len(x), -1, -1)
        query = query + self.scale_encoder(scale.to(x.dtype).reshape(-1, 1)).unsqueeze(
            1
        )
        x = torch.cat((query, x), dim=1)
        x = x + self.position[:, : x.shape[1]]
        return self.regressor(self.norm(self.encoder(x))[:, 0])


def selected_symbol_and_scale(h, log_mean, log_std, augment_phase):
    rms = root_mean_square(h)
    selected = h[..., 0]
    if augment_phase:
        phase = torch.rand(len(h), device=h.device) * (2 * torch.pi)
        selected = (
            selected * torch.polar(torch.ones_like(phase), phase)[:, None, None, None]
        )
    return selected, (rms.log() - log_mean) / log_std


def lwm_tokens(selected):
    values = torch.view_as_real(selected).float()
    scale = values.abs().flatten(1).amax(1).clamp_min(torch.finfo(values.dtype).eps)
    values = values / scale[:, None, None, None, None]
    batch, receivers, transmitters, frequencies, _ = values.shape
    spatial = receivers * transmitters
    if spatial % 4 or frequencies % 4:
        raise ValueError(f"LWM requires dimensions divisible by four, got {spatial}x{frequencies}")
    values = values.reshape(batch, spatial, frequencies, 2)
    patches = values.reshape(
        batch, spatial // 4, 4, frequencies // 4, 4, 2
    ).permute(0, 1, 3, 2, 4, 5).contiguous().reshape(batch, LWM_TOKEN_COUNT, 32)
    cls = torch.full((batch, 1, 32), 0.2, device=values.device)
    return torch.cat((cls, patches), 1)


def resize_lwm_position_state(state):
    key = "embedding.pos_embed.weight"
    old = state[key]
    if old.shape[0] - 1 != 4 * 18:
        raise ValueError(f"Expected a 4x18 LWM position grid, got {old.shape[0] - 1} tokens")
    spatial_blocks = (NUM_RX * NUM_TX) // 4
    frequency_blocks = NUM_SUBCARRIERS // 4
    cls = old[:1]
    grid = old[1:].reshape(4, 18, old.shape[1]).permute(2, 0, 1).unsqueeze(0)
    grid = F.interpolate(
        grid, size=(spatial_blocks, frequency_blocks),
        mode="bicubic", align_corners=False,
    )
    resized = dict(state)
    resized[key] = torch.cat(
        (cls, grid.squeeze(0).permute(1, 2, 0).reshape(LWM_TOKEN_COUNT, old.shape[1])),
        dim=0,
    )
    return resized


def contra_matrix(selected):
    matrix = selected.reshape(len(selected), NUM_RX * NUM_TX, NUM_SUBCARRIERS)
    return normalize_max_component(matrix)[0]


def interpolate_positions(values, length):
    values = values.detach().transpose(1, 2)
    return F.interpolate(values, size=length, mode="linear", align_corners=True).transpose(
        1, 2
    )


class FrozenPipeline(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.mode = args.mode
        self.is_lwm = args.mode.startswith("lwm11")
        self.use_rafc = args.mode.endswith("_rafc")
        if self.is_lwm:
            ckpt = torch.load(args.lwm_checkpoint, map_location="cpu", weights_only=True)
            state = ckpt.get("model", ckpt)
            state = {key.removeprefix("module."): value for key, value in state.items()}
            state = resize_lwm_position_state(state)
            self.backbone = LWM(
                element_length=32,
                d_model=128,
                n_layers=12,
                max_len=LWM_TOKEN_COUNT + 1,
                n_heads=8,
                dropout=0.1,
            )
            self.backbone.load_state_dict(state, strict=True)
            dim = 128
            token_count = LWM_TOKEN_COUNT
        else:
            ckpt = torch.load(
                args.contra_checkpoint, map_location="cpu", weights_only=True
            )
            self.backbone = ContraWiMAE(
                patch_size=(16, 1),
                encoder_dim=64,
                encoder_layers=12,
                encoder_nhead=16,
                decoder_layers=4,
                decoder_nhead=8,
                mask_ratio=0.9,
                contrastive_dim=64,
                temperature=0.2,
                snr_min=5.0,
                snr_max=40.0,
                device=torch.device("cpu"),
                max_len=144,
            )
            self.backbone.load_state_dict(ckpt["model_state_dict"], strict=True)
            position = self.backbone.encoder.positional_encoding
            position.position_embeddings = nn.Parameter(
                interpolate_positions(
                    position.position_embeddings, CONTRA_TOKEN_COUNT
                ),
                requires_grad=False,
            )
            dim = 64
            token_count = CONTRA_TOKEN_COUNT
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False
        self.fusion = RAFC(dim, activation="gelu") if self.use_rafc else None
        self.head = ScaleTransformerLocalizationHead(dim, max_len=token_count)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def frozen_features(self, selected):
        captured = {}
        with torch.no_grad():
            if self.is_lwm:
                output = self.backbone.embedding(lwm_tokens(selected))
                for idx, layer in enumerate(self.backbone.layers):
                    output, _ = layer(output)
                    if idx in LAYER_IDS:
                        captured[idx] = output[:, 1:]
                last = output[:, 1:]
            else:
                patches = self.backbone.patcher(contra_matrix(selected))
                output = self.backbone.encoder.linear_1(patches)
                output = self.backbone.encoder.positional_encoding(output)
                for idx, layer in enumerate(self.backbone.encoder.transformer.layers):
                    output = layer(output)
                    if idx in LAYER_IDS:
                        captured[idx] = output
                norm = self.backbone.encoder.transformer.norm
                last = norm(output) if norm is not None else output
        if self.fusion is None:
            return last, None
        return self.fusion((captured[0], captured[5], captured[11]))

    def forward(self, selected, scale):
        features, alpha = self.frozen_features(selected)
        return self.head(features, scale), alpha


def generate_city16(args):
    if args.data_source_root is None:
        raise ValueError("--data-source-root is required with --generate-cache")
    source_root = args.data_source_root.expanduser().resolve()
    if not (source_root / "deepmimo_scenarios").is_dir():
        raise FileNotFoundError(f"DeepMIMO scenarios not found under {source_root}")
    sys.path.insert(0, str(source_root))
    if "mlflow" not in sys.modules:
        try:
            __import__("mlflow")
        except ImportError:
            sys.modules["mlflow"] = types.ModuleType("mlflow")
    previous_directory = Path.cwd()
    try:
        os.chdir(source_root)
        import deepmimo as dm

        channel_parameters = dm.ChannelParameters()
        channel_parameters.bs_antenna.rotation = np.array([0.0, 0.0, -135.0])
        channel_parameters.bs_antenna.fov = np.array([360.0, 180.0])
        channel_parameters.bs_antenna.shape = np.array([NUM_TX, 1])
        channel_parameters.bs_antenna.spacing = 0.5
        channel_parameters.ue_antenna.rotation = np.array([0.0, 0.0, 0.0])
        channel_parameters.ue_antenna.fov = np.array([360.0, 180.0])
        channel_parameters.ue_antenna.shape = np.array([NUM_RX, 1])
        channel_parameters.ue_antenna.spacing = 0.5
        channel_parameters.freq_domain = True
        channel_parameters.num_paths = 25
        channel_parameters.ofdm.bandwidth = 120000.0 * NUM_SUBCARRIERS
        channel_parameters.ofdm.subcarriers = NUM_SUBCARRIERS
        channel_parameters.ofdm.selected_subcarriers = np.arange(NUM_SUBCARRIERS)
        channel_parameters.ofdm.rx_filter = 0
        environment = {
            "fc_Hz": 28e9,
            "sc_spacing": 120000.0,
            "user_speed_range": [0.1, 5.0],
        }
        rng = np.random.default_rng(args.seed)
        channels, positions, base_stations, receiver_ids = [], [], [], []
        scenario = "city_16_sanfrancisco_3p5"
        for station in range(3):
            index_path = source_root / "deepmimo_scenarios" / f"{scenario}_{station}_idx.npy"
            source_indices = np.load(index_path)
            sample_count = min(args.samples_per_bs, len(source_indices))
            selected = rng.choice(source_indices, size=sample_count, replace=False)
            dataset = dm.load(
                scenario, tx_sets={station + 1: [0]}, rx_sets={0: selected}
            )
            dataset.compute_channels_temporal(
                channel_parameters,
                environment["fc_Hz"],
                NUM_SYMBOLS,
                1.0 / environment["sc_spacing"],
                environment["user_speed_range"],
            )
            channel = np.asarray(dataset.channels)
            position = np.asarray(dataset.rx_pos, dtype=np.float32)
            expected = (sample_count, NUM_RX, NUM_TX, NUM_SUBCARRIERS, NUM_SYMBOLS)
            if channel.shape != expected:
                raise RuntimeError(f"Expected generated CSI {expected}, got {channel.shape}")
            print(
                f"DATA bs={station + 1} channel={channel.shape} position={position.shape}",
                flush=True,
            )
            channels.append(channel)
            positions.append(position)
            base_stations.append(
                np.full(sample_count, station + 1, dtype=np.int64)
            )
            receiver_ids.append(selected.astype(np.int64))
        return (
            torch.from_numpy(np.concatenate(channels)),
            torch.from_numpy(np.concatenate(positions)),
            torch.from_numpy(np.concatenate(base_stations)),
            torch.from_numpy(np.concatenate(receiver_ids)),
        )
    finally:
        os.chdir(previous_directory)


def save_or_load_city16(args):
    if not args.cache.exists():
        if not args.generate_cache:
            raise FileNotFoundError(
                f"Localization cache not found: {args.cache}. Use --generate-cache with --data-source-root or provide an existing cache."
            )
        args.cache.parent.mkdir(parents=True, exist_ok=True)
        tensors = generate_city16(args)
        temporary = args.cache.with_suffix(args.cache.suffix + f".{os.getpid()}.tmp")
        torch.save(tensors, temporary)
        os.replace(temporary, args.cache)
        print(f"CACHE saved={args.cache}", flush=True)
    print(f"CACHE load={args.cache}", flush=True)
    tensors = torch.load(
        args.cache, map_location="cpu", weights_only=True, mmap=True
    )
    if not isinstance(tensors, (tuple, list)) or len(tensors) != 4:
        raise ValueError("Localization cache must contain four tensors")
    channel, position, base_station, receiver_id = tensors
    expected_shape = (NUM_RX, NUM_TX, NUM_SUBCARRIERS, NUM_SYMBOLS)
    if channel.ndim != 5 or tuple(channel.shape[1:]) != expected_shape:
        raise ValueError(f"Expected channel [N,{','.join(map(str, expected_shape))}], got {channel.shape}")
    if position.shape != (len(channel), 3):
        raise ValueError(f"Expected position [N,3], got {position.shape}")
    if len(base_station) != len(channel) or len(receiver_id) != len(channel):
        raise ValueError("Localization metadata length mismatch")
    print(
        f"DATA cache={args.cache} samples={len(channel)} channel={tuple(channel.shape)} position={tuple(position.shape)}",
        flush=True,
    )
    return tensors


def make_loader(ds, indices, batch_size, workers, shuffle=False, seed=None):
    return DataLoader(
        Subset(ds, indices.tolist()),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed) if seed is not None else None,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def evaluate(
    model,
    loader,
    device,
    pos_min,
    pos_range,
    log_mean,
    log_std,
    criterion,
    save_rows=False,
):
    model.eval()
    total_loss = total_dist = count = 0
    rows = []
    errors = []
    with torch.no_grad():
        for h, target, bs, ue in loader:
            h = h.to(device, non_blocking=True)
            target_gpu = target.to(device, non_blocking=True)
            selected, scale = selected_symbol_and_scale(h, log_mean, log_std, False)
            pred_norm, alpha = model(selected, scale)
            pred = pred_norm * pos_range + pos_min
            err = torch.linalg.vector_norm(pred - target_gpu, dim=1)
            total_loss += criterion(
                pred_norm, (target_gpu - pos_min) / pos_range
            ).item() * len(h)
            total_dist += err.sum().item()
            count += len(h)
            if save_rows:
                pred_cpu, err_cpu = pred.cpu(), err.cpu()
                errors.extend(err_cpu.tolist())
                for i in range(len(h)):
                    a = (
                        alpha[i].cpu().tolist()
                        if alpha is not None
                        else [float("nan")] * 3
                    )
                    rows.append(
                        [
                            int(bs[i]),
                            int(ue[i]),
                            *target[i].tolist(),
                            *pred_cpu[i].tolist(),
                            float(err_cpu[i]),
                            *a,
                        ]
                    )
    return (
        total_loss / count,
        total_dist / count,
        rows,
        np.asarray(errors, dtype=np.float32),
    )


def main():
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    tensors = save_or_load_city16(args)
    if args.prepare_only:
        return
    h_all, pos, bs_ids, ue_ids = tensors
    tr, va, te = split_indices(len(h_all), args.seed)
    pos_min_cpu = pos[tr].min(0).values
    pos_range_cpu = (pos[tr].max(0).values - pos_min_cpu).clamp_min(1e-6)
    log_parts = []
    for chunk in tr.split(512):
        x = h_all[chunk]
        log_parts.append(root_mean_square(x).log())
    log_rms = torch.cat(log_parts)
    log_mean_cpu = log_rms.mean()
    log_std_cpu = log_rms.std().clamp_min(1e-6)
    ds = TensorDataset(h_all, pos, bs_ids, ue_ids)
    device = torch.device(args.device)
    pos_min, pos_range, log_mean, log_std = [
        x.to(device) for x in (pos_min_cpu, pos_range_cpu, log_mean_cpu, log_std_cpu)
    ]
    model = FrozenPipeline(args).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-3)
    criterion = nn.SmoothL1Loss(beta=0.05)
    start_epoch, best, stale, history = 1, math.inf, 0, []
    best_path = args.output / f"{args.mode}_best.pt"
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=True)
        if model.fusion is not None:
            model.fusion.load_state_dict(state["fusion"])
        model.head.load_state_dict(state["head"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch, best = state["epoch"] + 1, state["best_val_mean_m"]
    val_loader = make_loader(ds, va, args.batch_size, args.num_workers)
    test_loader = make_loader(ds, te, args.batch_size, args.num_workers)
    if args.eval_only:
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required with --eval-only")
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        state = state.get("trainable", state)
        fusion_state = state.get("fusion", state.get("router"))
        if model.fusion is not None:
            if fusion_state is None:
                raise RuntimeError("RAFC state is missing from the checkpoint")
            model.fusion.load_state_dict(fusion_state)
        model.head.load_state_dict(state["head"])
        loss, _, rows, errors = evaluate(
            model, test_loader, device, pos_min, pos_range,
            log_mean, log_std, criterion,
        )
        print(json.dumps({
            "mode": args.mode,
            "checkpoint": str(args.checkpoint),
            "test_loss": loss,
            "metrics": metrics(errors),
            "first_prediction": rows[0] if rows else None,
        }, indent=2))
        return
    config = {
        **vars(args),
        "train": len(tr),
        "val": len(va),
        "test": len(te),
        "input_shape": [len(h_all), NUM_RX, NUM_TX, NUM_SUBCARRIERS, NUM_SYMBOLS],
        "ofdm_symbol": 0,
        "awgn": "none",
        "train_augmentation": "global_phase_rotation",
        "backbone": "frozen",
        "layers": list(LAYER_IDS) if model.use_rafc else [11],
        "feature_tokens": LWM_TOKEN_COUNT if model.is_lwm else CONTRA_TOKEN_COUNT,
        "position_embedding": "linearly interpolated from the frozen pretrained table",
        "normalization": "native per-symbol max + standardized full-sample log RMS scale token",
        "pos_min": pos_min_cpu.tolist(),
        "pos_max": (pos_min_cpu + pos_range_cpu).tolist(),
        "log_rms_mean": log_mean_cpu.item(),
        "log_rms_std": log_std_cpu.item(),
    }
    with open(args.output / f"{args.mode}_config.json", "w") as f:
        json.dump(
            {k: str(v) if isinstance(v, Path) else v for k, v in config.items()},
            f,
            indent=2,
        )
    print(
        f"START mode={args.mode} device={device} seed={args.seed} trainable={sum(p.numel() for p in trainable):,} train={len(tr)} val={len(va)} test={len(te)}",
        flush=True,
    )
    started = time.time()
    for epoch in range(start_epoch, args.epochs + 1):
        train_loader = make_loader(
            ds, tr, args.batch_size, args.num_workers, True, args.seed + epoch
        )
        model.train()
        loss_sum = dist_sum = count = 0
        alpha_sum = torch.zeros(3, device=device)
        for h, target, _, _ in train_loader:
            h = h.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            selected, scale = selected_symbol_and_scale(h, log_mean, log_std, True)
            optimizer.zero_grad(set_to_none=True)
            pred_norm, alpha = model(selected, scale)
            loss = criterion(pred_norm, (target - pos_min) / pos_range)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            b = len(h)
            loss_sum += loss.item() * b
            dist_sum += (
                torch.linalg.vector_norm(
                    pred_norm * pos_range + pos_min - target, dim=1
                )
                .sum()
                .item()
            )
            count += b
            if alpha is not None:
                alpha_sum += alpha.detach().sum(0)
        val_loss, val_dist, _, _ = evaluate(
            model, val_loader, device, pos_min, pos_range, log_mean, log_std, criterion
        )
        row = {
            "epoch": epoch,
            "train_loss": loss_sum / count,
            "train_mean_m": dist_sum / count,
            "val_loss": val_loss,
            "val_mean_m": val_dist,
            "alpha_low": (
                (alpha_sum / count)[0].item() if model.use_rafc else float("nan")
            ),
            "alpha_mid": (
                (alpha_sum / count)[1].item() if model.use_rafc else float("nan")
            ),
            "alpha_high": (
                (alpha_sum / count)[2].item() if model.use_rafc else float("nan")
            ),
        }
        history.append(row)
        if val_dist < best:
            best, stale = val_dist, 0
            torch.save(
                {
                    "fusion": model.fusion.state_dict() if model.fusion else None,
                    "head": model.head.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_mean_m": best,
                },
                best_path,
            )
        else:
            stale += 1
        print("EPOCH " + args.mode + " " + json.dumps(row), flush=True)
        if stale >= args.patience:
            print(
                f"EARLY_STOP mode={args.mode} epoch={epoch} best={best:.6f}", flush=True
            )
            break
    state = torch.load(best_path, map_location=device, weights_only=True)
    if model.fusion is not None:
        model.fusion.load_state_dict(state["fusion"])
    model.head.load_state_dict(state["head"])
    _, _, rows, errors = evaluate(
        model,
        test_loader,
        device,
        pos_min,
        pos_range,
        log_mean,
        log_std,
        criterion,
        True,
    )
    result = {
        "variant": args.mode,
        "seed": args.seed,
        "best_epoch": state["epoch"],
        "best_val_mean_m": best,
        **metrics(errors),
        "elapsed_s": time.time() - started,
    }
    with open(args.output / f"{args.mode}_history.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=history[0].keys())
        w.writeheader()
        w.writerows(history)
    with open(args.output / f"{args.mode}_predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "bs",
                "ue_id",
                "true_x",
                "true_y",
                "true_z",
                "pred_x",
                "pred_y",
                "pred_z",
                "error_m",
                "alpha_low",
                "alpha_mid",
                "alpha_high",
            ]
        )
        w.writerows(rows)
    np.savez_compressed(args.output / f"{args.mode}_paired_errors.npz", error_m=errors)
    with open(args.output / f"{args.mode}_metrics.json", "w") as f:
        json.dump(result, f, indent=2)
    print("RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
