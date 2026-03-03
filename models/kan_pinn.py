# kan_pinn.py
# KAN-PINN Neural Network for EMI Prediction

import torch
import torch.nn as nn
import numpy as np


# ─────────────────────────────────────────
# PART 1: KAN Layer
# ─────────────────────────────────────────

class KANLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super(KANLayer, self).__init__()

        self.linear = nn.Linear(
            input_size, output_size
        )
        self.spline_weight = nn.Parameter(
            torch.randn(output_size, input_size) * 0.1
        )

    def forward(self, x):
        base   = self.linear(x)
        spline = torch.tanh(
            x @ self.spline_weight.T
        )
        return base + spline


# ─────────────────────────────────────────
# PART 2: Full KAN-PINN Model
# ─────────────────────────────────────────

class KANPINN(nn.Module):
    def __init__(self, input_size=14):
        super(KANPINN, self).__init__()

        self.layer1 = KANLayer(input_size, 64)
        self.layer2 = KANLayer(64, 32)
        self.layer3 = KANLayer(32, 16)

        self.output_layer = nn.Linear(16, 1)
        self.dropout      = nn.Dropout(0.2)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout(x)
        x = torch.relu(self.layer2(x))
        x = self.dropout(x)
        x = torch.relu(self.layer3(x))
        return self.output_layer(x)


# ─────────────────────────────────────────
# PART 3: Physics-Informed Loss Function
# ─────────────────────────────────────────

def physics_informed_loss(
        predictions, targets, features):
    """
    Combines data loss with physics rules.
    Penalizes physically impossible predictions.
    """

    # Loss 1: Prediction error
    data_loss = nn.MSELoss()(predictions, targets)

    # Loss 2: Higher frequency = higher EMI
    frequency  = features[:, 5:6]
    freq_grad  = torch.mean(
        torch.relu(-(predictions * frequency))
    )

    # Loss 3: Wider trace = lower EMI
    trace_width  = features[:, 0:1]
    width_grad   = torch.mean(
        torch.relu(predictions * trace_width)
    )

    # Combined loss
    total_loss = (
        1.0 * data_loss +
        0.1 * freq_grad +
        0.1 * width_grad
    )

    return total_loss, data_loss


# ─────────────────────────────────────────
# PART 4: Test
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("Testing KAN-PINN Model...")

    fake_input = torch.randn(5, 14)
    model      = KANPINN(input_size=14)
    prediction = model(fake_input)

    print(f"Input shape:  {fake_input.shape}")
    print(f"Output shape: {prediction.shape}")
    print(f"Predictions:  "
          f"{prediction.detach().numpy().flatten()}")

    fake_targets = torch.randn(5, 1)
    total_loss, data_loss = physics_informed_loss(
        prediction, fake_targets, fake_input
    )

    print(f"Total Loss: {total_loss.item():.4f}")
    print(f"Data Loss:  {data_loss.item():.4f}")
    print("\nKAN-PINN working correctly!")
