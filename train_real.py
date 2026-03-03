# train_real.py
# Retrains KAN-PINN on real physics data

import torch
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

sys.path.append('data')
sys.path.append('models')
sys.path.append('utils')

from graph_builder import build_pcb_graph, graph_to_feature_vector
from kan_pinn import KANPINN, physics_informed_loss

# ─────────────────────────────────────────
# STEP 1: Load Dataset
# ─────────────────────────────────────────

print("=" * 55)
print("  NeuroShield-PCB: Real Data Retraining")
print("=" * 55)

print("\n[1/6] Loading real physics dataset...")
df    = pd.read_csv('data/real_pcb_dataset.csv')
LIMIT = 40.0

print(f"      Loaded {len(df)} samples")
print(f"      EMI range: {df['emi_dbm'].min():.1f}"
      f" to {df['emi_dbm'].max():.1f} dBm")

passing = (df['emi_dbm'] < LIMIT).sum()
failing = (df['emi_dbm'] >= LIMIT).sum()
print(f"      Passing: {passing} ({passing/10:.1f}%)")
print(f"      Failing: {failing} ({failing/10:.1f}%)")

# ─────────────────────────────────────────
# STEP 2: Convert to Graph Features
# ─────────────────────────────────────────

print("\n[2/6] Converting to graph features...")

feature_vectors = []
skipped         = 0

for i in range(len(df)):
    row = df.iloc[i]
    pcb_row = {
        'trace_width_mm':     float(row['trace_width_mm']),
        'trace_length_mm':    float(row['trace_length_mm']),
        'ground_distance_mm': float(row['ground_distance_mm']),
        'stitching_vias':     float(row['stitching_vias']),
        'decap_distance_mm':  float(row['decap_distance_mm']),
        'frequency_mhz':      float(row['frequency_mhz']),
    }
    try:
        G  = build_pcb_graph(pd.Series(pcb_row))
        fv = graph_to_feature_vector(G)
        feature_vectors.append(fv)
    except Exception:
        skipped += 1

print(f"      Converted: {len(feature_vectors)}")
if skipped > 0:
    print(f"      Skipped:   {skipped}")

X          = np.array(feature_vectors)
y          = df['emi_dbm'].values[:len(feature_vectors)]
input_size = X.shape[1]

print(f"      Shape: {X.shape}")
print(f"      Input size: {input_size}")

# ─────────────────────────────────────────
# STEP 3: Normalize
# ─────────────────────────────────────────

print("\n[3/6] Normalizing...")

X_mean = X.mean(axis=0)
X_std  = X.std(axis=0) + 1e-8
X_norm = (X - X_mean) / X_std

y_mean = y.mean()
y_std  = y.std() + 1e-8
y_norm = (y - y_mean) / y_std

print(f"      Y mean: {y_mean:.2f} | Y std: {y_std:.2f}")

os.makedirs('outputs', exist_ok=True)
np.save('outputs/X_mean.npy', X_mean)
np.save('outputs/X_std.npy',  X_std)
np.save('outputs/y_mean.npy', np.array([y_mean]))
np.save('outputs/y_std.npy',  np.array([y_std]))
print("      Normalization params saved!")

# ─────────────────────────────────────────
# STEP 4: Split Dataset
# ─────────────────────────────────────────

print("\n[4/6] Splitting dataset...")

n         = len(X_norm)
train_end = int(0.70 * n)
val_end   = int(0.85 * n)

X_train = X_norm[:train_end]
X_val   = X_norm[train_end:val_end]
X_test  = X_norm[val_end:]
y_train = y_norm[:train_end]
y_val   = y_norm[train_end:val_end]
y_test  = y_norm[val_end:]
y_test_real = y[val_end:]

print(f"      Train:      {len(X_train)} (70%)")
print(f"      Validation: {len(X_val)} (15%)")
print(f"      Test:       {len(X_test)} (15%)")

X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.FloatTensor(y_train).reshape(-1, 1)
X_val_t   = torch.FloatTensor(X_val)
y_val_t   = torch.FloatTensor(y_val).reshape(-1, 1)
X_test_t  = torch.FloatTensor(X_test)
y_test_t  = torch.FloatTensor(y_test).reshape(-1, 1)

# ─────────────────────────────────────────
# STEP 5: Train
# ─────────────────────────────────────────

print("\n[5/6] Training...")
print("-" * 45)

model     = KANPINN(input_size=input_size)
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-5
)

# Fixed: removed verbose argument
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    patience=10,
    factor=0.5
)

EPOCHS         = 200
best_val_loss  = float('inf')
best_weights   = None
patience_count = 0
PATIENCE       = 20
train_losses   = []
val_losses     = []

for epoch in range(EPOCHS):

    # Train
    model.train()
    optimizer.zero_grad()
    preds       = model(X_train_t)
    loss, _     = physics_informed_loss(
        preds, y_train_t, X_train_t
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), 1.0
    )
    optimizer.step()
    train_losses.append(loss.item())

    # Validate
    model.eval()
    with torch.no_grad():
        val_preds   = model(X_val_t)
        val_loss, _ = physics_informed_loss(
            val_preds, y_val_t, X_val_t
        )
        val_losses.append(val_loss.item())

    scheduler.step(val_loss)

    # Save best
    if val_loss.item() < best_val_loss:
        best_val_loss  = val_loss.item()
        best_weights   = {
            k: v.clone()
            for k, v in model.state_dict().items()
        }
        patience_count = 0
    else:
        patience_count += 1

    # Early stopping
    if patience_count >= PATIENCE:
        print(f"\n  Early stopping at epoch {epoch+1}")
        break

    if (epoch + 1) % 20 == 0:
        print(f"  Epoch [{epoch+1:3d}/{EPOCHS}] | "
              f"Train: {loss.item():.4f} | "
              f"Val: {val_loss.item():.4f}")

model.load_state_dict(best_weights)
print(f"\n  Best val loss: {best_val_loss:.4f}")

# ─────────────────────────────────────────
# STEP 6: Evaluate and Save
# ─────────────────────────────────────────

print("\n[6/6] Evaluating...")

model.eval()
with torch.no_grad():
    test_preds_norm = model(X_test_t).numpy().flatten()

# Denormalize
test_preds_real = test_preds_norm * y_std + y_mean

mae    = np.mean(np.abs(test_preds_real - y_test_real))
rmse   = np.sqrt(np.mean(
    (test_preds_real - y_test_real) ** 2
))

pred_pass      = test_preds_real < LIMIT
real_pass      = y_test_real < LIMIT
compliance_acc = np.mean(pred_pass == real_pass) * 100

print(f"\n  MAE:            {mae:.4f} dBm")
print(f"  RMSE:           {rmse:.4f} dBm")
print(f"  Compliance Acc: {compliance_acc:.1f}%")

print(f"\n  {'Predicted':>12} {'Actual':>10} "
      f"{'Error':>8} {'Pass?':>6}")
print(f"  {'-'*40}")

for i in range(8):
    pred   = test_preds_real[i]
    actual = y_test_real[i]
    error  = abs(pred - actual)
    status = 'PASS' if pred < LIMIT else 'FAIL'
    print(f"  {pred:>12.2f} {actual:>10.2f} "
          f"{error:>8.2f} {status:>6}")

# Save model
torch.save({
    'model_state_dict': best_weights,
    'input_size':       input_size,
    'trained_on':       'real_physics_data',
    'mae_dbm':          float(mae),
    'compliance_acc':   float(compliance_acc)
}, 'outputs/kan_pinn_model.pth')

print(f"\n  Model saved!")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(train_losses, label='Train',
             color='blue', linewidth=1.5)
axes[0].plot(val_losses, label='Validation',
             color='orange', linewidth=1.5)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('Training Progress')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].scatter(y_test_real, test_preds_real,
                alpha=0.5, color='blue', s=20)
mn = min(y_test_real.min(), test_preds_real.min())
mx = max(y_test_real.max(), test_preds_real.max())
axes[1].plot([mn, mx], [mn, mx],
             'r--', linewidth=2,
             label='Perfect prediction')
axes[1].axhline(y=LIMIT, color='orange',
                linestyle='--',
                label='CISPR Limit')
axes[1].axvline(x=LIMIT, color='orange',
                linestyle='--')
axes[1].set_xlabel('Actual EMI (dBm)')
axes[1].set_ylabel('Predicted EMI (dBm)')
axes[1].set_title('Predicted vs Actual')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/real_training_results.png',
            dpi=150, bbox_inches='tight')
print(f"  Plot saved!")

print("\n" + "=" * 55)
print("  Retraining Complete!")
print(f"  Accuracy:       ±{mae:.2f} dBm")
print(f"  Compliance Acc: {compliance_acc:.1f}%")
print("=" * 55)
