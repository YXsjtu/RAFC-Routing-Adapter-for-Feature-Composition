"""Representation-adaptive feature fusion."""

import torch
import torch.nn as nn


class RAFC(nn.Module):
    def __init__(self, dimension, feature_count=3, reduction=4, activation="relu"):
        super().__init__()
        hidden = max(dimension // reduction, 1)
        self.feature_count = feature_count
        activations = {"relu": nn.ReLU(inplace=True), "gelu": nn.GELU()}
        if activation not in activations:
            raise ValueError(f"Unsupported RAFC activation: {activation}")
        self.router = nn.Sequential(
            nn.Linear(feature_count * dimension, hidden),
            activations[activation],
            nn.Linear(hidden, feature_count),
        )

    def forward(self, features):
        features = (
            list(features.values()) if isinstance(features, dict) else list(features)
        )
        if len(features) != self.feature_count:
            raise ValueError(
                f"Expected {self.feature_count} features, got {len(features)}"
            )
        pooled = torch.cat([feature.mean(dim=1) for feature in features], dim=-1)
        weights = self.router(pooled).softmax(dim=-1)
        fused = sum(
            weights[:, index, None, None] * feature
            for index, feature in enumerate(features)
        )
        return fused, weights
