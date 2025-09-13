import argparse
import os
from dataclasses import dataclass
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import yaml

from models.perception_model import PerceptionModel


@dataclass
class Config:
    sampling_rate_hz: int
    sequence_length: int
    classes: list
    model: Dict[str, Any]
    loss_weights: Dict[str, float]


class NpzDataset(Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=True)
        self.X = data["X"]  # [N, T, 8]
        self.y_ex = data["y_ex"]  # [N]
        self.y_form = data["y_form"]  # [N]
        self.y_rep = data["y_rep"]  # [N, T]
        self.y_fatigue = data["y_fatigue"]  # [N]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx]).float().transpose(0, 1)  # [8, T]
        return {
            "x": x,
            "y_ex": torch.tensor(self.y_ex[idx], dtype=torch.long),
            "y_form": torch.tensor(self.y_form[idx], dtype=torch.long),
            "y_rep": torch.from_numpy(self.y_rep[idx]).float(),
            "y_fatigue": torch.tensor(self.y_fatigue[idx], dtype=torch.float32),
        }


def train_one_epoch(model, loader, optim, device, loss_w):
    model.train()
    ce = nn.CrossEntropyLoss()
    bce = nn.BCELoss()
    mse = nn.MSELoss()

    total = 0.0
    for batch in loader:
        x = batch["x"].to(device)
        y_ex = batch["y_ex"].to(device)
        y_form = batch["y_form"].to(device)
        y_rep = batch["y_rep"].to(device)
        y_fatigue = batch["y_fatigue"].to(device)

        out = model(x)
        loss = (
            loss_w["exercise_ce"] * ce(out["exercise_logits"], y_ex)
            + loss_w["form_ce"] * ce(out["form_logits"], y_form)
            + loss_w["rep_bce"] * bce(out["rep_probs"], y_rep)
            + loss_w["fatigue_mse"] * mse(out["fatigue_score"], y_fatigue)
            + loss_w["vq_commit"] * out["vq_loss"]
        )

        optim.zero_grad()
        loss.backward()
        optim.step()

        total += loss.item()
    return total / max(1, len(loader))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data", required=True, help="Path to .npz dataset")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

    ds = NpzDataset(args.data)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0)

    model = PerceptionModel(
        input_channels=8,
        hidden_dim=cfg["model"]["hidden_dim"],
        codebook_size=cfg["model"]["codebook_size"],
        num_exercises=len(cfg["classes"]),
        num_forms=4,
        transformer_layers=cfg["model"]["transformer_layers"],
        nhead=cfg["model"]["transformer_heads"],
    ).to(device)

    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    os.makedirs(args.out + "/checkpoints", exist_ok=True)

    for ep in range(1, args.epochs + 1):
        loss = train_one_epoch(model, loader, optim, device, cfg["loss_weights"])
        print(f"epoch {ep}: loss={loss:.4f}")
        torch.save(model.state_dict(), f"{args.out}/checkpoints/last.pth")


if __name__ == "__main__":
    main()
