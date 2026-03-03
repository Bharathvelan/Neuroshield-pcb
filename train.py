# train.py
# This script trains our KAN-PINN model on PCB data

import torch
import torch.optim as optim
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# Tell Python where to find our files
sys.path.append('data')
sys.path.append('models')
sys.path.append('utils')

from sample_board import generate_pcb_samples
from graph_builder import build_pcb_graph, graph_to_feature_vector
from kan_pinn import KANPINN, physics_informed_loss

# ─────────────────────────────────────────
# STEP 1: Prepare the Data
# ─────────────────────────────────────────

print("=" * 50)
print("  NeuroShield-PCB Training Started!")
print("=" * 50)

print("\n[1/5] Generating PCB samples...")
df = generate_pcb_samples(1000)
print(f"      Generated {len(df)} PCB designs")

# Convert each PCB design into a feature vector
print("\n[2/5] Converting PCBs to graph features...")
feature_vectors = []
for i in range(len(df)):
    row = df.iloc[i]
    G = build_pcb_graph(row)
    fv = graph_to_feature_vector(G)
    feature_vectors.append(fv)

# Stack all feature vectors into one big matrix
X = np.array(feature_vectors)
y = df['emi_dbm'].values

# Auto-detect input size (fixes the shape mismatch!)
input_size = X.shape[1]

print(f"      Feature matrix shape: {X.shape}")
print(f"      Target vector shape:  {y.shape}")
print(f"      Auto-detected input size: {input_size}")

# ─────────────────────────────────────────
# STEP 2: Split into Train and Test Sets
# ─────────────────────────────────────────

print("\n[3/5] Splitting data into train/test sets...")

split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"      Training samples: {len(X_train)}")
print(f"      Testing samples:  {len(X_test)}")

# Convert to PyTorch tensors
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train).reshape(-1, 1)
X_test_t  = torch.FloatTensor(X_test)
y_test_t  = torch.FloatTensor(y_test).reshape(-1, 1)

# ─────────────────────────────────────────
# STEP 3: Create Model and Optimizer
# ─────────────────────────────────────────

print("\n[4/5] Setting up model and optimizer...")

# Use auto-detected input size
model     = KANPINN(input_size=input_size)
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(
    optimizer, step_size=50, gamma=0.5
)

print(f"      Model input size: {input_size}")
print(f"      Model parameters: "
      f"{sum(p.numel() for p in model.parameters())}")

# ─────────────────────────────────────────
# STEP 4: Training Loop
# ─────────────────────────────────────────

print("\n[5/5] Training the model...")
print("-" * 50)

EPOCHS = 100
train_losses = []
test_losses  = []

for epoch in range(EPOCHS):

    # --- Training phase ---
    model.train()
    optimizer.zero_grad()

    predictions = model(X_train_t)

    total_loss, data_loss = physics_informed_loss(
        predictions, y_train_t, X_train_t
    )

    total_loss.backward()
    optimizer.step()
    scheduler.step()

    train_losses.append(total_loss.item())

    # --- Testing phase ---
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test_t)
        test_loss, _ = physics_informed_loss(
            test_predictions, y_test_t, X_test_t
        )
        test_losses.append(test_loss.item())

    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"  Epoch [{epoch+1:3d}/{EPOCHS}] | "
              f"Train Loss: {total_loss.item():.4f} | "
              f"Test Loss:  {test_loss.item():.4f}")

# ─────────────────────────────────────────
# STEP 5: Save the Trained Model
# ─────────────────────────────────────────

os.makedirs('outputs', exist_ok=True)

# Save model + input size together
torch.save({
    'model_state_dict': model.state_dict(),
    'input_size': input_size
}, 'outputs/kan_pinn_model.pth')

print("\n✅ Model saved to outputs/kan_pinn_model.pth")

# ─────────────────────────────────────────
# STEP 6: Plot Training Progress
# ─────────────────────────────────────────

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss', color='blue')
plt.plot(test_losses,  label='Test Loss',  color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('NeuroShield-PCB: Training Progress')
plt.legend()
plt.grid(True)
plt.savefig('outputs/training_plot.png')
print("✅ Training plot saved to outputs/training_plot.png")

# ─────────────────────────────────────────
# STEP 7: Final Accuracy Check
# ─────────────────────────────────────────

model.eval()
with torch.no_grad():
    final_predictions = model(X_test_t).numpy().flatten()
    actual_values     = y_test_t.numpy().flatten()

    mae = np.mean(np.abs(final_predictions - actual_values))
    print(f"\n✅ Final Mean Absolute Error: {mae:.4f} dBm")
    print("\n   Sample Predictions vs Actual:")
    print(f"   {'Predicted':>12}  {'Actual':>10}  {'Error':>8}")
    print(f"   {'-'*35}")
    for i in range(5):
        error = abs(final_predictions[i] - actual_values[i])
        print(f"   {final_predictions[i]:>12.2f}"
              f"  {actual_values[i]:>10.2f}"
              f"  {error:>8.2f}")

print("\n" + "=" * 50)
print("  Training Complete!")
print("=" * 50)
