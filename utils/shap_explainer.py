# shap_explainer.py
# Explains WHY our model made a prediction
# Tells engineers exactly which PCB feature
# is causing the most EMI problems

import shap
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append('data')
sys.path.append('models')
sys.path.append('utils')

from sample_board import generate_pcb_samples
from graph_builder import build_pcb_graph, graph_to_feature_vector
from kan_pinn import KANPINN

# ─────────────────────────────────────────
# PART 1: Load the trained model
# ─────────────────────────────────────────

def load_model():
    """Loads our previously trained KAN-PINN model"""

    checkpoint = torch.load(
        'outputs/kan_pinn_model.pth',
        weights_only=False
    )
    input_size = checkpoint['input_size']
    model = KANPINN(input_size=input_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, input_size


# ─────────────────────────────────────────
# PART 2: Wrapper so SHAP can use our model
# ─────────────────────────────────────────

def model_predict(X_numpy):
    """
    SHAP needs a function it can call.
    This wraps our PyTorch model so SHAP
    can pass numpy arrays to it.
    """
    X_tensor = torch.FloatTensor(X_numpy)
    with torch.no_grad():
        predictions = model(X_tensor)
    return predictions.numpy().flatten()


# ─────────────────────────────────────────
# PART 3: Generate Explanation Report
# ─────────────────────────────────────────

def explain_pcb(pcb_features, feature_names):
    """
    Takes a PCB's features and explains
    which ones are causing the most EMI.
    """

    print("\n" + "=" * 55)
    print("  NeuroShield-PCB: EMI Root Cause Analysis")
    print("=" * 55)

    # Predict EMI for this PCB
    prediction = model_predict(pcb_features.reshape(1, -1))[0]
    print(f"\n  Predicted EMI Level: {prediction:.2f} dBm")

    # CISPR 32 Class B limit (simplified)
    limit = 40.0
    if prediction > limit:
        margin = prediction - limit
        print(f"  Status: ❌ FAIL — {margin:.2f} dBm over limit!")
    else:
        margin = limit - prediction
        print(f"  Status: ✅ PASS — {margin:.2f} dBm below limit")

    # ── Run SHAP analysis ──
    print("\n  Analyzing root causes...")

    # Use a small background dataset for SHAP
    background = torch.FloatTensor(X_background)
    explainer = shap.KernelExplainer(
        model_predict,
        X_background[:50]  # use 50 samples as background
    )

    # Calculate SHAP values for our PCB
    shap_values = explainer.shap_values(
        pcb_features.reshape(1, -1),
        nsamples=100
    )

    shap_vals = shap_values[0]

    # ── Print ranked report ──
    print("\n  EMI Contribution by Feature:")
    print(f"  {'Rank':<6} {'Feature':<25} "
          f"{'Value':>10} {'EMI Impact':>12}")
    print(f"  {'-'*55}")

    # Sort features by absolute SHAP value
    sorted_idx = np.argsort(np.abs(shap_vals))[::-1]

    fixes = {
        'trace_length_mm':    'Shorten the trace or add ground stitching',
        'frequency_mhz':      'Add EMI filter or ferrite bead on trace',
        'decap_distance_mm':  'Move decoupling capacitor closer to IC',
        'ground_distance_mm': 'Reduce layer spacing or add ground pour',
        'stitching_vias':     'Add more stitching vias near trace',
        'trace_width_mm':     'Increase trace width to reduce impedance',
    }

    print("\n  🔍 Top 3 Root Causes & Recommended Fixes:")
    print(f"  {'-'*55}")

    for rank, idx in enumerate(sorted_idx[:3]):
        if idx < len(feature_names):
            fname  = feature_names[idx]
            fval   = pcb_features[idx]
            impact = shap_vals[idx]
            fix    = fixes.get(fname, 'Review this parameter')

            print(f"\n  #{rank+1} Feature:  {fname}")
            print(f"     Value:    {fval:.3f}")
            print(f"     Impact:   {impact:+.4f} dBm")
            print(f"     Fix:      {fix}")

    # ── Save SHAP bar chart ──
    os.makedirs('outputs', exist_ok=True)

    # Only use feature-length shap values
    valid_len   = min(len(shap_vals), len(feature_names))
    plot_shap   = shap_vals[:valid_len]
    plot_names  = feature_names[:valid_len]
    plot_vals   = pcb_features[:valid_len]

    colors = ['red' if s > 0 else 'blue' for s in plot_shap]

    plt.figure(figsize=(10, 6))
    bars = plt.barh(plot_names, plot_shap, color=colors)
    plt.xlabel('SHAP Value (Impact on EMI prediction)')
    plt.title('NeuroShield-PCB: EMI Root Cause Analysis\n'
              '(Red = increases EMI, Blue = reduces EMI)')
    plt.axvline(x=0, color='black', linewidth=0.8)
    plt.tight_layout()
    plt.savefig('outputs/shap_explanation.png')
    print(f"\n  ✅ SHAP chart saved to outputs/shap_explanation.png")
    print("=" * 55)

    return shap_vals


# ─────────────────────────────────────────
# PART 4: Run the explainer
# ─────────────────────────────────────────

if __name__ == "__main__":

    # Load trained model
    print("Loading trained model...")
    model, input_size = load_model()

    # Generate test data
    df = generate_pcb_samples(200)

    feature_vectors = []
    for i in range(len(df)):
        row = df.iloc[i]
        G   = build_pcb_graph(row)
        fv  = graph_to_feature_vector(G)
        feature_vectors.append(fv)

    X_all = np.array(feature_vectors)

    # Background dataset for SHAP
    X_background = X_all[:100]

    # Pick one PCB to explain (the first test PCB)
    test_pcb = X_all[100]

    # Feature names for the report
    feature_names = [
        'trace_width_mm',
        'trace_length_mm',
        'frequency_mhz',
        'ground_distance_mm',
        'stitching_vias',
        'decap_distance_mm',
        'node_type_trace',
        'node_type_ground',
        'node_type_via',
        'node_type_decap',
        'coupling_ground',
        'coupling_via',
        'coupling_decap',
        'coupling_gnd_via'
    ]

    # Run explanation
    explain_pcb(test_pcb, feature_names)
