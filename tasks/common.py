"""Shared downstream Transformer layers."""

from pathlib import Path

import torch
import torch.nn as nn


def resolve_data_path(data_root, path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else Path(data_root).expanduser() / path


def downstream_state(model):
    state = {"head": model.head.state_dict()}
    router = getattr(model, "router", None)
    state["router"] = router.state_dict() if router is not None else None
    return state


def load_downstream_state(model, state):
    model.head.load_state_dict(state["head"])
    router = getattr(model, "router", None)
    if router is not None:
        if state.get("router") is None:
            raise RuntimeError("RAFC state is missing from the downstream checkpoint")
        router.load_state_dict(state["router"])


class PositionalEncoding(nn.Module):
    def __init__(self, dimension, max_length=1024):
        super().__init__()
        encoding = torch.zeros(max_length, dimension)
        position = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(
            torch.arange(0, dimension, 2, dtype=torch.float32)
            * (-torch.log(torch.tensor(10000.0)) / dimension)
        )
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, inputs):
        return inputs + self.encoding[:, : inputs.shape[1]]


class TransformerHead(nn.Module):
    def __init__(
        self,
        input_dimension,
        output_dimension,
        model_dimension=256,
        heads=8,
        layers=2,
        feedforward_dimension=512,
        dropout=0.1,
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_dimension, model_dimension)
        self.position = PositionalEncoding(model_dimension)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dimension,
            nhead=heads,
            dim_feedforward=feedforward_dimension,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.output_projection = nn.Linear(model_dimension, output_dimension)

    def forward(self, inputs):
        hidden = self.position(self.input_projection(inputs))
        return self.output_projection(self.encoder(hidden))
