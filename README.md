# quantumleap-imu-engine

Perception data engine and model for the Chimera Ascendant PoC.

- Low-impact exercise set: standing_march, arm_circles, step_touch, wall_pushup, seated_extension
- Phone-only signals: 50 Hz IMU + 2-channel barometer (pressure kPa, altitude m)
- CNN + (stub VQ) + Transformer backbone with multi-task heads
- No Weights & Biases. Simple CLI and logs to stdout.

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Generate a tiny synthetic dataset

```
python -m data_engine.synthetic_generator --config configs/default.yml --out data/synth_small.npz --samples 1000
```

## Train a quick smoke model (CPU ok)

```
python -m training.trainer --config configs/default.yml --data data/synth_small.npz --epochs 1
```

## Export Core ML (optional, macOS only)

```
python -m export.coreml_export --checkpoint out/checkpoints/last.pth --output out/ChimeraPerception.mlmodel
```

## Config
See `configs/default.yml` for placement invariance ranges and dataset sizing.
