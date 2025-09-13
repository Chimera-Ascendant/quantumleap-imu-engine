import argparse
import os
import time
from typing import Tuple, Dict

import numpy as np
import yaml


EXERCISES = [
    "standing_march",
    "arm_circles",
    "step_touch",
    "wall_pushup",
    "seated_extension",
]

FORMS = [
    "perfect",
    "balance_unstable",
    "range_too_low",
    "jerk_high",
]


def _rand_orientation(cfg: Dict[str, Tuple[float, float]]):
    yaw = np.deg2rad(np.random.uniform(*cfg["yaw_deg_range"]))
    pitch = np.deg2rad(np.random.uniform(*cfg["pitch_deg_range"]))
    roll = np.deg2rad(np.random.uniform(*cfg["roll_deg_range"]))
    return yaw, pitch, roll


def _apply_orientation(accel: np.ndarray, yaw: float, pitch: float, roll: float) -> np.ndarray:
    # Simple rotation around z (yaw), y (pitch), x (roll)
    cz, sz = np.cos(yaw), np.sin(yaw)
    cy, sy = np.cos(pitch), np.sin(pitch)
    cx, sx = np.cos(roll), np.sin(roll)

    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    R = Rz @ Ry @ Rx
    return accel @ R.T


def _exercise_pattern(name: str, n: int, fs: int) -> Tuple[np.ndarray, np.ndarray]:
    t = np.linspace(0, (n - 1) / fs, n)
    # Base gravity vector in z
    accel = np.zeros((n, 3))
    accel[:, 2] = 9.81
    gyro = np.zeros((n, 3))

    if name == "standing_march":
        f = 1.2
        accel[:, 1] += 0.8 * np.sin(2 * np.pi * f * t)
        accel[:, 2] += 0.3 * np.abs(np.sin(2 * np.pi * f * t))
        gyro[:, 0] += 0.2 * np.cos(2 * np.pi * f * t)
    elif name == "arm_circles":
        f = 0.8
        accel[:, 0] += 0.6 * np.cos(2 * np.pi * f * t)
        accel[:, 1] += 0.6 * np.sin(2 * np.pi * f * t)
        gyro[:, 2] += 0.15 * np.sin(2 * np.pi * 2 * f * t)
    elif name == "step_touch":
        f = 1.0
        accel[:, 1] += 0.5 * np.sign(np.sin(2 * np.pi * f * t))
        accel[:, 2] += 0.2 * (np.sin(2 * np.pi * f * t) > 0).astype(float)
        gyro[:, 1] += 0.1 * np.sin(2 * np.pi * f * t)
    elif name == "wall_pushup":
        f = 0.5
        accel[:, 0] += 0.3 * np.sin(2 * np.pi * f * t)
        accel[:, 2] += 0.2 * np.cos(2 * np.pi * f * t)
        gyro[:, 0] += 0.1 * np.sin(2 * np.pi * f * t)
    elif name == "seated_extension":
        f = 0.7
        accel[:, 1] += 0.4 * np.sin(2 * np.pi * f * t)
        gyro[:, 0] += 0.1 * np.cos(2 * np.pi * f * t)
    else:
        pass

    return accel, gyro


def _inject_form_error(accel: np.ndarray, form: str, cfg: Dict) -> np.ndarray:
    noisy = accel.copy()
    if form == "balance_unstable":
        jitter = np.random.normal(0, 0.2, size=noisy.shape)
        noisy += jitter
    elif form == "range_too_low":
        noisy[:, 1] *= 0.6
        noisy[:, 2] *= 0.6
    elif form == "jerk_high":
        spikes = (np.random.rand(*noisy.shape) < 0.02) * np.random.normal(0, 1.2, size=noisy.shape)
        noisy += spikes
    return noisy


def _barometer_stream(n: int) -> np.ndarray:
    pressure_kpa = np.full(n, 101.325) + np.random.normal(0, 0.003, n)
    altitude_m = np.random.normal(0, 0.05, n)
    return np.stack([pressure_kpa, altitude_m], axis=1)


def _rep_boundaries(pattern: np.ndarray) -> np.ndarray:
    # Simple zero-crossing as proxy for rep peaks
    s = pattern
    zc = (np.diff(np.signbit(s)) != 0).astype(float)
    zc = np.pad(zc, (1, 0))
    return zc


def generate_dataset(config_path: str, out_path: str, samples: int):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    fs = int(cfg.get("sampling_rate_hz", 50))
    seq_len = int(cfg.get("sequence_length", 300))

    X = np.zeros((samples, seq_len, 8), dtype=np.float32)  # 6 IMU + 2 baro
    y_ex = np.zeros((samples,), dtype=np.int64)
    y_form = np.zeros((samples,), dtype=np.int64)
    y_rep = np.zeros((samples, seq_len), dtype=np.float32)
    y_fatigue = np.zeros((samples,), dtype=np.float32)

    for i in range(samples):
        ex = np.random.choice(EXERCISES)
        frm = np.random.choice(FORMS, p=[0.6, 0.15, 0.15, 0.10])
        fatigue = np.random.beta(2, 5)

        accel, gyro = _exercise_pattern(ex, seq_len, fs)

        # Tempo drift & velocity drop
        drift_pct = np.random.uniform(*cfg["fatigue_simulation"]["tempo_drift_pct_range"]) if fatigue > 0.6 else 0.0
        if drift_pct > 0:
            # Slow down by resampling index mapping
            t_idx = np.linspace(0, seq_len - 1, seq_len)
            t_slow = np.linspace(0, (seq_len - 1) * (1 - 0.2 * drift_pct), seq_len)
            accel = np.vstack([
                np.interp(t_idx, t_slow, accel[:, 0], left=accel[0, 0], right=accel[-1, 0]),
                np.interp(t_idx, t_slow, accel[:, 1], left=accel[0, 1], right=accel[-1, 1]),
                np.interp(t_idx, t_slow, accel[:, 2], left=accel[0, 2], right=accel[-1, 2]),
            ]).T
            gyro = np.vstack([
                np.interp(t_idx, t_slow, gyro[:, 0], left=gyro[0, 0], right=gyro[-1, 0]),
                np.interp(t_idx, t_slow, gyro[:, 1], left=gyro[0, 1], right=gyro[-1, 1]),
                np.interp(t_idx, t_slow, gyro[:, 2], left=gyro[0, 2], right=gyro[-1, 2]),
            ]).T

        if fatigue > 0.6:
            accel *= (1.0 - np.random.uniform(*cfg["fatigue_simulation"]["velocity_drop_pct_range"]) * 0.2)
            accel += np.random.normal(0, np.random.uniform(*cfg["fatigue_simulation"]["jitter_gain_range"]) * 0.1, size=accel.shape)

        # Apply form error
        accel = _inject_form_error(accel, frm, cfg)

        # Randomized orientation
        yaw, pitch, roll = _rand_orientation(cfg["placement_invariance"])
        accel = _apply_orientation(accel, yaw, pitch, roll)

        baro = _barometer_stream(seq_len)

        # Stack channels: accel(xyz), gyro(xyz), baro(pressure, altitude)
        imu = np.concatenate([accel, gyro], axis=1)
        X[i] = np.concatenate([imu, baro], axis=1).astype(np.float32)
        y_ex[i] = EXERCISES.index(ex)
        y_form[i] = FORMS.index(frm)

        # Rep boundaries using accel y pattern as proxy
        y_rep[i] = _rep_boundaries(accel[:, 1]).astype(np.float32)
        y_fatigue[i] = fatigue

        if (i + 1) % max(1, samples // 10) == 0:
            print(f"Generated {i+1}/{samples}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    np.savez_compressed(
        out_path,
        X=X,
        y_ex=y_ex,
        y_form=y_form,
        y_rep=y_rep,
        y_fatigue=y_fatigue,
        classes=np.array(EXERCISES),
        forms=np.array(FORMS),
        fs=np.array([fs]),
        seq_len=np.array([seq_len]),
        timestamp_ms=int(time.time() * 1000),
    )
    print(f"Saved dataset to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=1000)
    args = ap.parse_args()
    generate_dataset(args.config, args.out, args.samples)
