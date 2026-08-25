"""DeepMIMO-v4-beta temporal channel recipe used by the experiments.

DeepMIMO did not expose temporal OFDM rendering at the pinned revision. The
local routines below are the small extension used for this project's data.
DeepMIMO itself remains an external Apache-2.0 dependency.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

import deepmimo as dm
from deepmimo import consts as c
from deepmimo.generator.channel import ChannelParameters, OFDM_PathGenerator


PINNED_DEEPMIMO_COMMIT = "fb7584bc90bf9f2bf693107a67821c42bde71e08"
INDEX_ROOT = Path(__file__).resolve().parent / "indices"
SAMPLES_PER_BS = 2000

PHYSICAL_PROFILES = {
    "array_profiles": [
        {"name": "ula4_ula4", "probability": 0.60, "bs_shape": [4, 1],
         "ue_shape": [4, 1], "spacing_lambda": 0.5},
        {"name": "upa2_ula4", "probability": 0.20, "bs_shape": [2, 2],
         "ue_shape": [4, 1], "spacing_lambda": 0.5},
        {"name": "upa2_upa2", "probability": 0.20, "bs_shape": [2, 2],
         "ue_shape": [2, 2], "spacing_lambda": 0.5},
    ],
    "path_profiles": [
        {"name": "paths_10", "num_paths": 10, "probability": 0.20},
        {"name": "paths_15", "num_paths": 15, "probability": 0.30},
        {"name": "paths_25", "num_paths": 25, "probability": 0.50},
    ],
    "mobility_profiles": [
        {"name": "pedestrian", "probability": 0.35, "speed_mps": [0.0, 2.0]},
        {"name": "urban", "probability": 0.40, "speed_mps": [2.0, 15.0]},
        {"name": "vehicular", "probability": 0.25, "speed_mps": [15.0, 35.0]},
    ],
    "orientation": {
        "bs_yaw_jitter_deg": [-15.0, 15.0],
        "ue_yaw_deg": [0.0, 360.0],
        "ue_pitch_deg": [-10.0, 10.0],
    },
}


def stable_seed(seed: int, *parts) -> int:
    payload = ":".join(map(str, (seed, *parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


def configure_scenario_root(scenario_root: Path) -> None:
    scenario_root.mkdir(parents=True, exist_ok=True)
    dm.config.set("scenarios_folder", str(scenario_root.resolve()))


def make_channel_parameters(num_subcarriers: int) -> ChannelParameters:
    params = ChannelParameters()
    params.bs_antenna.rotation = np.array([0.0, 0.0, -135.0])
    params.bs_antenna.fov = np.array([360.0, 180.0])
    params.bs_antenna.shape = np.array([4, 1])
    params.bs_antenna.spacing = 0.5
    params.ue_antenna.rotation = np.array([0.0, 0.0, 0.0])
    params.ue_antenna.fov = np.array([360.0, 180.0])
    params.ue_antenna.shape = np.array([4, 1])
    params.ue_antenna.spacing = 0.5
    params.freq_domain = True
    params.num_paths = 25
    params.ofdm.bandwidth = 30_000.0 * num_subcarriers
    params.ofdm.subcarriers = num_subcarriers
    params.ofdm.selected_subcarriers = np.arange(num_subcarriers)
    params.ofdm.rx_filter = 0
    return params


def spherical_to_cartesian(azimuth, elevation):
    return np.stack(
        [np.sin(elevation) * np.cos(azimuth),
         np.sin(elevation) * np.sin(azimuth),
         np.cos(elevation)],
        axis=-1,
    )


def _temporal_channel_kernel(array_product, powers, delays, phases, dopplers,
                             ofdm_params, temporal_phase, freq_domain=True):
    sample_period = 1 / ofdm_params[c.PARAMSET_OFDM_BANDWIDTH]
    subcarriers = ofdm_params[c.PARAMSET_OFDM_SC_SAMP]
    path_generator = OFDM_PathGenerator(ofdm_params, subcarriers)
    n_users, max_paths = powers.shape
    n_rx, n_tx = array_product.shape[1:3]
    last_dimension = len(subcarriers) if freq_domain else max_paths
    channel = np.zeros(
        (n_users, n_rx, n_tx, last_dimension, temporal_phase.shape[2]),
        dtype=np.complex64,
    )
    valid_masks = ~np.isnan(powers)
    for index in tqdm(range(n_users), desc="Generating temporal channels", leave=False):
        valid = valid_masks[index]
        if not valid.any():
            continue
        array = array_product[index][..., valid]
        power = powers[index, valid]
        delay = delays[index, valid]
        phase = phases[index, valid]
        doppler = dopplers[index, valid]
        time_phase = temporal_phase[index, valid, :]
        if freq_domain:
            path_gain = path_generator.generate(
                pwr=power, toa=delay, phs=phase, Ts=sample_period,
                dopplers=doppler,
            ).T
            channel[index] = np.sum(
                array[:, :, None, None, :]
                * path_gain[None, None, :, None, :]
                * time_phase.T[None, None, None, :, :],
                axis=-1,
            )
        else:
            gain = np.sqrt(power) * np.exp(
                1j * (np.deg2rad(phase) + 2 * np.pi * doppler)
            )
            channel[index, ..., :len(power)] = array * gain[None, None, :]
    return channel


def compute_temporal_channels(dataset, params, symbols: int, speed_range,
                              velocity_seed: int):
    """Render one base station deterministically with the legacy extension."""
    dataset.set_channel_params(params)
    array_product = dataset._compute_array_response_product()
    n_paths = min(params.num_paths, dataset.delay.shape[-1])
    directions = spherical_to_cartesian(
        dataset[c.AOA_AZ_ROT_PARAM_NAME][..., :n_paths],
        dataset[c.AOA_EL_ROT_PARAM_NAME][..., :n_paths],
    )
    rng = np.random.RandomState(velocity_seed)
    velocity_azimuth = rng.uniform(0, 2 * np.pi, size=dataset.n_ue)
    speed = rng.uniform(*speed_range, size=dataset.n_ue)
    velocity = np.stack(
        [np.cos(velocity_azimuth) * speed,
         np.sin(velocity_azimuth) * speed,
         np.zeros_like(speed)],
        axis=-1,
    )
    temporal_doppler = (3_500_000_000.0 / 3e8) * np.sum(
        velocity[:, None, :] * directions, axis=-1
    )
    symbol_times = np.arange(symbols) * (1 / 30_000.0)
    temporal_phase = np.exp(
        1j * 2 * np.pi * temporal_doppler[:, :, None]
        * symbol_times[None, None, :]
    )
    channel = _temporal_channel_kernel(
        array_product[..., :n_paths],
        dataset._power_linear_ant_gain[..., :n_paths],
        dataset.delay[..., :n_paths],
        dataset.phase[..., :n_paths],
        np.zeros((dataset.n_ue, n_paths)),
        params.ofdm,
        temporal_phase,
        params.freq_domain,
    )
    return channel


def load_base_station(scene: str, base_station: int):
    full_name = f"{scene}_3p5"
    index_path = INDEX_ROOT / f"{full_name}_{base_station}_idx.npy"
    index_manifest = json.loads((INDEX_ROOT / "manifest.json").read_text())
    expected_hash = index_manifest["sha256"][index_path.name]
    actual_hash = hashlib.sha256(index_path.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise RuntimeError(f"Receiver-index checksum mismatch: {index_path}")
    indices = np.load(index_path)
    if indices.shape != (SAMPLES_PER_BS,):
        raise RuntimeError(f"Unexpected receiver-index shape: {index_path}: {indices.shape}")
    return dm.load(
        full_name,
        tx_sets={base_station + 1: [0]},
        rx_sets={0: indices},
    )


def _sample_profiles(rng, profiles, count):
    probabilities = np.asarray([item["probability"] for item in profiles])
    probabilities = probabilities / probabilities.sum()
    return rng.choice(len(profiles), size=count, p=probabilities)


def compute_diverse_channels(dataset, base_params, seed: int, group_name: str):
    profiles = PHYSICAL_PROFILES
    rng = np.random.default_rng(stable_seed(seed, group_name))
    count = dataset.n_ue
    array_ids = _sample_profiles(rng, profiles["array_profiles"], count)
    path_ids = _sample_profiles(rng, profiles["path_profiles"], count)
    mobility_ids = _sample_profiles(rng, profiles["mobility_profiles"], count)
    ue_yaw = rng.uniform(*profiles["orientation"]["ue_yaw_deg"], size=count)
    ue_pitch = rng.uniform(*profiles["orientation"]["ue_pitch_deg"], size=count)
    tx_position = np.asarray(dataset.tx_pos).reshape(-1, 3)[0]
    delta = np.asarray(dataset.rx_pos).mean(axis=0) - tx_position
    centroid_yaw = np.degrees(np.arctan2(delta[1], delta[0]))
    result = None
    combinations = np.stack([array_ids, path_ids, mobility_ids], axis=1)
    for array_id, path_id, mobility_id in np.unique(combinations, axis=0):
        selected = np.flatnonzero(
            (array_ids == array_id) & (path_ids == path_id)
            & (mobility_ids == mobility_id)
        )
        subset = dataset.subset(selected)
        params = base_params.deepcopy()
        array_profile = profiles["array_profiles"][int(array_id)]
        path_profile = profiles["path_profiles"][int(path_id)]
        mobility = profiles["mobility_profiles"][int(mobility_id)]
        params.bs_antenna.shape = np.asarray(array_profile["bs_shape"])
        params.ue_antenna.shape = np.asarray(array_profile["ue_shape"])
        params.bs_antenna.spacing = array_profile["spacing_lambda"]
        params.ue_antenna.spacing = array_profile["spacing_lambda"]
        params.num_paths = path_profile["num_paths"]
        params.bs_antenna.rotation = np.asarray([
            0.0, 0.0,
            centroid_yaw + rng.uniform(
                *profiles["orientation"]["bs_yaw_jitter_deg"]
            ),
        ])
        params.ue_antenna.rotation = np.column_stack(
            [np.zeros(len(selected)), ue_pitch[selected], ue_yaw[selected]]
        )
        channels = compute_temporal_channels(
            subset, params, symbols=14, speed_range=mobility["speed_mps"],
            velocity_seed=stable_seed(
                seed, group_name, int(array_id), int(path_id), int(mobility_id)
            ),
        )
        if result is None:
            result = np.empty((count, *channels.shape[1:]), dtype=channels.dtype)
        result[selected] = channels
    return result


def render_scene_to_file(scene: str, output: Path, num_subcarriers: int,
                         symbols: int, seed: int, diverse: bool,
                         overwrite: bool = False):
    expected = (3 * SAMPLES_PER_BS, 4, 4, num_subcarriers, symbols)
    if output.exists() and not overwrite:
        cached = np.load(output, mmap_mode="r")
        if cached.shape != expected or cached.dtype != np.complex64:
            raise RuntimeError(f"Invalid existing cache {output}: {cached.shape}, {cached.dtype}")
        print(f"[reuse] {output} {cached.shape}")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    target = np.lib.format.open_memmap(
        output, mode="w+", dtype=np.complex64, shape=expected
    )
    base_params = make_channel_parameters(num_subcarriers)
    for base_station in range(3):
        print(f"[render] {scene} BS {base_station + 1}/3")
        dataset = load_base_station(scene, base_station)
        if diverse:
            channel = compute_diverse_channels(
                dataset, base_params, seed, f"{scene}:bs{base_station + 1}"
            )
        else:
            channel = compute_temporal_channels(
                dataset, base_params, symbols, [5.0, 5.0],
                stable_seed(seed, scene, f"bs{base_station + 1}", "velocity"),
            )
        permutation = np.random.RandomState(seed).permutation(len(channel))
        start = base_station * SAMPLES_PER_BS
        target[start:start + SAMPLES_PER_BS] = channel[permutation]
        target.flush()
        del channel, dataset
    del target


def dft_codebook(num_tx: int, num_beams: int = 64):
    spatial_frequency = -1.0 + 2.0 * np.arange(num_beams) / num_beams
    antenna = np.arange(num_tx)[:, None]
    codebook = np.exp(1j * np.pi * antenna * spatial_frequency[None, :]) / np.sqrt(num_tx)
    return codebook.astype(np.complex64), spatial_frequency.astype(np.float32)


def beam_labels(csi, codebook, batch_size=256):
    labels = np.empty(len(csi), dtype=np.int64)
    best = np.empty(len(csi), dtype=np.float32)
    second = np.empty(len(csi), dtype=np.float32)
    for start in range(0, len(csi), batch_size):
        stop = min(start + batch_size, len(csi))
        channel = np.asarray(csi[start:stop], dtype=np.complex64)
        covariance = np.einsum(
            "brtfs,brufs->btu", channel.conj(), channel, optimize=True
        ) / np.float32(channel.shape[1] * channel.shape[3] * channel.shape[4])
        power = np.einsum(
            "tk,btu,uk->bk", codebook.conj(), covariance, codebook,
            optimize=True,
        ).real
        pair_indices = np.argpartition(power, kth=-2, axis=1)[:, -2:]
        rows = np.arange(stop - start)
        pair_power = power[rows[:, None], pair_indices]
        order = np.argsort(pair_power, axis=1)
        second_index = pair_indices[rows, order[:, 0]]
        best_index = pair_indices[rows, order[:, 1]]
        labels[start:stop] = best_index
        best[start:stop] = power[rows, best_index]
        second[start:stop] = power[rows, second_index]
    return labels, best, second
