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

    class Wrapper(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, x):  # x: [B, 8, T]
            out = self.base(x)
            # Return deterministic tuple for Core ML
            return (
                out["exercise_logits"],  # [B, num_classes]
                out["form_logits"],      # [B, num_forms]
                out["rep_probs"],        # [B, T]
                out["fatigue_score"],    # [B]
            )

    base = PerceptionModel()
    sd = torch.load(checkpoint, map_location="cpu")
    base.load_state_dict(sd)
    base.eval()
    model = Wrapper(base)
    model.eval()

    example = torch.randn(1, 8, sequence_length)
    # Avoid torch trace graph-diff checks due to squeeze ops by disabling check_trace
    ts_module = torch.jit.trace(model, example, check_trace=False)

    mlmodel = ct.convert(
        ts_module,
        inputs=[ct.TensorType(name="sensor_data", shape=example.shape)],
        outputs=[
            ct.TensorType(name="exercise_logits"),
            ct.TensorType(name="form_logits"),
            ct.TensorType(name="rep_probs"),
            ct.TensorType(name="fatigue_score"),
        ],
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        minimum_deployment_target=ct.target.iOS15,
    )

    mlmodel.short_description = "Chimera Perception Engine"
    if output.endswith(".mlmodel"):
        print("[warn] ML Program requires .mlpackage; changing extension automatically.")
        output = output[:-8] + ".mlpackage"
    mlmodel.save(output)
    print(f"Saved Core ML model to {output}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seq", type=int, default=300)
    args = ap.parse_args()
    export_coreml(args.checkpoint, args.output, args.seq)
