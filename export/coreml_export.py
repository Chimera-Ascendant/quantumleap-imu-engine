import argparse
import os

import torch

from models.perception_model import PerceptionModel


def export_coreml(checkpoint: str, output: str, sequence_length: int = 300):
    try:
        import coremltools as ct
    except Exception as e:
        raise RuntimeError(
            "coremltools is required for export. Please install coremltools on macOS."
        ) from e

    model = PerceptionModel()
    sd = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(sd)
    model.eval()

    example = torch.randn(1, 8, sequence_length)
    traced = torch.jit.trace(model, example)

    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="sensor_data", shape=example.shape)],
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS15,
    )

    mlmodel.short_description = "Chimera Perception Engine"
    mlmodel.save(output)
    print(f"Saved Core ML model to {output}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seq", type=int, default=300)
    args = ap.parse_args()
    export_coreml(args.checkpoint, args.output, args.seq)
