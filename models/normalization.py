"""Normalization rules shared by the downstream experiments."""

import torch


def root_mean_square(channel, observed_only=False):
    dimensions = tuple(range(1, channel.ndim))
    power = channel.abs().square()
    if not observed_only:
        return power.mean(dim=dimensions).clamp_min(1e-24).sqrt()
    observed = channel.ne(0)
    count = observed.sum(dim=dimensions).clamp_min(1)
    return (
        (power * observed).sum(dim=dimensions).div(count).clamp_min(1e-24).sqrt()
    )
def normalize_max_component(channel):
    components = torch.view_as_real(channel)
    scale = components.abs().flatten(1).amax(dim=1)
    scale = scale.clamp_min(torch.finfo(components.dtype).eps)
    view = scale.reshape(-1, *([1] * (channel.ndim - 1)))
    return channel / view, scale
