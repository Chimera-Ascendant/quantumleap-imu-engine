import torch
from models.perception_model import PerceptionModel

def test_forward_shape():
    model = PerceptionModel()
    x = torch.randn(2, 8, 300)
    out = model(x)
    assert out["exercise_logits"].shape == (2, 5)
    assert out["form_logits"].shape == (2, 4)
    assert out["rep_probs"].shape == (2, 300)
    assert out["fatigue_score"].shape == (2,)
