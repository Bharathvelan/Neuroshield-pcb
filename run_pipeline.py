# run_pipeline.py
# Runs the complete NeuroShield-PCB pipeline:
# Step 1: Load trained model
# Step 2: Analyze a PCB design
# Step 3: Explain root causes
# Step 4: Optimize the layout
# Step 5: Generate final report

import torch
import numpy as np
import pandas as pd
import json
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.append('data')
sys.path.append('models')
sys.path.append('utils')

from sample_board import generate_pcb_samples
from graph_builder import build_pcb_graph, graph_to_feature_vector
from kan_pinn import KANPINN
from drl_optimizer import calculate_emi, optimize_pcb

os.makedirs('outputs', exist_ok=True)


# ─────────────────────────────────────────
# HELPER: Get EMI prediction
# ─────────────────────────────────────────

def get_emi(pcb_dict, model, input_size):
    return calculate_emi(pcb_dict, model, input_size)


# ─────────────────────────────────────────
# STEP 1: Load Model
# ─────────────────────────────────────────

def step1_load_model():
    print("\n" + "=" * 60)
    print("  STEP 1: Loading Trained KAN-PINN Model")
    print("=" * 60)

    checkpoint = torch.load(
        'outputs/kan_pinn_model.pth',
        weights_only=False
    )
    input_size = checkpoint['input_size']
    model      = KANPINN(input_size=input_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print(f"  Model loaded successfully!")
    print(f"  Input size:   {input_size}")
    print(f"  Parameters:   "
          f"{sum(p.numel() for p in model.parameters())}")

    return model, input_size


# ─────────────────────────────────────────
# STEP 2: Analyze PCB
# ─────────────────────────────────────────

def step2_analyze(pcb, model, input_size):
    print("\n" + "=" * 60)
    print("  STEP 2: Analyzing PCB Design")
    print("=" * 60)

    emi   = get_emi(pcb, model, input_size)
    LIMIT = 40.0

    print(f"\n  PCB Parameters:")
    for key, val in pcb.items():
        print(f"    {key:<25} {val}")

    print(f"\n  Predicted EMI:  {emi:.2f} dBm")
    print(f"  CISPR 32 Limit: {LIMIT:.2f} dBm")

    if emi > LIMIT:
        print(f"  Status:         FAIL "
              f"({emi - LIMIT:.2f} dBm over limit)")
    else:
        print(f"  Status:         PASS "
              f"({LIMIT - emi:.2f} dBm below limit)")

    return emi


# ─────────────────────────────────────────
# STEP 3: Root Cause Analysis
# ─────────────────────────────────────────

def step3_root_cause(pcb, emi):
    print("\n" + "=" * 60)
    print("  STEP 3: Root Cause Analysis")
    print("=" * 60)

    # Physics-based importance scoring
    # Each score shows how much this parameter
    # contributes to EMI based on known physics rules
    scores = {
        'trace_length_mm':    pcb['trace_length_mm'] / 100.0,
        'frequency_mhz':      pcb['frequency_mhz'] / 3000.0,
        'decap_distance_mm':  pcb['decap_distance_mm'] / 15.0,
        'ground_distance_mm': pcb['ground_distance_mm'] / 2.0,
        'stitching_vias':     1.0 - pcb['stitching_vias'] / 10.0,
        'trace_width_mm':     1.0 - pcb['trace_width_mm'] / 2.0,
    }

    fixes = {
        'trace_length_mm':
            'Shorten trace or add ground stitching vias',
        'frequency_mhz':
            'Add ferrite bead or EMI filter on trace',
        'decap_distance_mm':
            'Move decoupling capacitor closer to IC pin',
        'ground_distance_mm':
            'Reduce layer spacing or add ground copper pour',
        'stitching_vias':
            'Add more stitching vias near signal trace',
        'trace_width_mm':
            'Increase trace width to reduce impedance',
    }

    sorted_causes = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    print(f"\n  Top EMI Root Causes:")
    print(f"  {'Rank':<6} {'Parameter':<25} "
          f"{'Score':>6}  Fix")
    print(f"  {'-'*70}")

    for rank, (param, score) in enumerate(sorted_causes[:3]):
        fix = fixes[param]
        print(f"  #{rank+1:<5} {param:<25} "
              f"{score:>6.3f}  {fix}")

    return scores, sorted_causes


# ─────────────────────────────────────────
# STEP 4: Optimize
# ─────────────────────────────────────────

def step4_optimize(pcb, model, input_size):
    print("\n" + "=" * 60)
    print("  STEP 4: Running Optimizer")
    print("=" * 60)

    best_pcb, best_emi, initial_emi, history = optimize_pcb(
        pcb, model, input_size,
        iterations=2000
    )

    return best_pcb, best_emi, history


# ─────────────────────────────────────────
# STEP 5: Generate Report
# ─────────────────────────────────────────

def step5_report(initial_pcb, optimized_pcb,
                 initial_emi, optimized_emi,
                 scores, sorted_causes, history):

    print("\n" + "=" * 60)
    print("  STEP 5: Generating Final Report")
    print("=" * 60)

    LIMIT       = 40.0
    improvement = initial_emi - optimized_emi

    # ── Create 4-panel figure ──
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        'NeuroShield-PCB — Full Analysis Report',
        fontsize=16, fontweight='bold', y=0.98
    )

    gs = gridspec.GridSpec(2, 2, figure=fig,
                           hspace=0.4, wspace=0.35)

    # ── Panel 1: EMI Comparison Bar Chart ──
    ax1 = fig.add_subplot(gs[0, 0])
    categories = ['Initial PCB', 'Optimized PCB', 'CISPR Limit']
    values     = [initial_emi, optimized_emi, LIMIT]
    colors     = ['red', 'green', 'orange']
    bars = ax1.bar(categories, values, color=colors,
                   alpha=0.8, edgecolor='black')
    ax1.axhline(y=LIMIT, color='orange',
                linestyle='--', linewidth=2)
    for bar, val in zip(bars, values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f'{val:.1f}', ha='center',
            fontweight='bold', fontsize=11
        )
    ax1.set_ylabel('EMI (dBm)')
    ax1.set_title('EMI Before vs After Optimization')
    ax1.set_ylim(0, max(values) * 1.2)

    # ── Panel 2: Root Cause Bar Chart ──
    ax2 = fig.add_subplot(gs[0, 1])
    params = [s[0].replace('_', '\n') for s in sorted_causes]
    vals   = [s[1] for s in sorted_causes]
    colors2 = plt.cm.RdYlGn_r(
        np.linspace(0.1, 0.9, len(params))
    )
    ax2.barh(params, vals, color=colors2,
             edgecolor='black', alpha=0.8)
    ax2.set_xlabel('EMI Impact Score')
    ax2.set_title('Root Cause Analysis\n(Higher = More EMI Impact)')
    ax2.axvline(x=0, color='black', linewidth=0.8)

    # ── Panel 3: Optimization Progress ──
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(history, color='blue',
             linewidth=1.5, label='Best EMI')
    ax3.axhline(y=LIMIT, color='red',
                linestyle='--', linewidth=2,
                label=f'Limit ({LIMIT} dBm)')
    ax3.axhline(y=initial_emi, color='orange',
                linestyle=':', linewidth=1.5,
                label=f'Initial ({initial_emi:.1f} dBm)')
    ax3.fill_between(
        range(len(history)), history, LIMIT,
        where=[h > LIMIT for h in history],
        alpha=0.15, color='red'
    )
    ax3.fill_between(
        range(len(history)), history, LIMIT,
        where=[h <= LIMIT for h in history],
        alpha=0.15, color='green'
    )
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Best EMI (dBm)')
    ax3.set_title('Optimization Progress')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    # ── Panel 4: Parameter Changes Table ──
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis('off')

    param_labels = [
        'Trace Width (mm)',
        'Trace Length (mm)',
        'Ground Dist (mm)',
        'Stitching Vias',
        'Decap Dist (mm)',
        'Frequency (MHz)',
    ]
    param_keys = [
        'trace_width_mm',
        'trace_length_mm',
        'ground_distance_mm',
        'stitching_vias',
        'decap_distance_mm',
        'frequency_mhz',
    ]

    table_data = []
    for label, key in zip(param_labels, param_keys):
        before  = float(initial_pcb.get(key, 0))
        after   = float(optimized_pcb.get(key, 0))
        changed = 'YES' if abs(before - after) > 0.01 else '-'
        table_data.append([label,
                           f'{before:.1f}',
                           f'{after:.1f}',
                           changed])

    # Add EMI rows
    table_data.append(['--- ', '---', '---', '---'])
    table_data.append(['EMI (dBm)',
                       f'{initial_emi:.2f}',
                       f'{optimized_emi:.2f}',
                       f'-{improvement:.2f}'])
    table_data.append(['Status',
                       'FAIL' if initial_emi > LIMIT
                       else 'PASS',
                       'PASS' if optimized_emi < LIMIT
                       else 'FAIL',
                       ''])

    table = ax4.table(
        cellText=table_data,
        colLabels=['Parameter', 'Before', 'After', 'Changed'],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.1, 1.6)

    # Color the EMI rows
    for j in range(4):
        table[len(table_data), j].set_facecolor('#ffcccc')
        table[len(table_data)-1, j].set_facecolor('#ccffcc')

    ax4.set_title('Parameter Changes Summary',
                  fontweight='bold', pad=15)

    plt.savefig('outputs/final_report.png',
                dpi=150, bbox_inches='tight')
    print(f"\n  Report saved to outputs/final_report.png")

    # ── Save JSON report ──
    clean_pcb = {
        k: float(v) for k, v in optimized_pcb.items()
    }
    report = {
        'initial_pcb':    {
            k: float(v) for k, v in initial_pcb.items()
        },
        'optimized_pcb':  clean_pcb,
        'initial_emi':    float(initial_emi),
        'optimized_emi':  float(optimized_emi),
        'improvement_db': float(improvement),
        'passes_cispr32': bool(optimized_emi < LIMIT),
        'top_causes': [
            {'parameter': s[0], 'score': float(s[1])}
            for s in sorted_causes[:3]
        ]
    }

    with open('outputs/final_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  JSON report saved to outputs/final_report.json")

    # ── Print summary ──
    print(f"\n  {'='*50}")
    print(f"  FINAL SUMMARY")
    print(f"  {'='*50}")
    print(f"  Initial EMI:   {initial_emi:.2f} dBm  "
          f"({'FAIL' if initial_emi > LIMIT else 'PASS'})")
    print(f"  Optimized EMI: {optimized_emi:.2f} dBm  "
          f"({'PASS' if optimized_emi < LIMIT else 'FAIL'})")
    print(f"  Improvement:   {improvement:.2f} dBm")

    if optimized_emi < LIMIT:
        print(f"\n  SUCCESS! Board passes CISPR 32!")
    else:
        print(f"\n  Progress made: {improvement:.2f} dBm reduced")
        print(f"  Run with more iterations for full compliance")

    print(f"  {'='*50}")


# ─────────────────────────────────────────
# MAIN: Run Full Pipeline
# ─────────────────────────────────────────

if __name__ == "__main__":

    print("\n")
    print("  ###########################################")
    print("  #      NeuroShield-PCB PIPELINE           #")
    print("  #  KAN-PINN + Optimizer + SHAP + Report   #")
    print("  ###########################################")

    # Define test PCB
    test_pcb = {
        'trace_width_mm':     0.2,
        'trace_length_mm':    85.0,
        'ground_distance_mm': 1.4,
        'stitching_vias':     1.0,
        'decap_distance_mm':  11.0,
        'frequency_mhz':      500.0
    }

    # Run all steps
    model, input_size = step1_load_model()
    initial_emi       = step2_analyze(
                            test_pcb, model, input_size)
    scores, causes    = step3_root_cause(test_pcb, initial_emi)
    opt_pcb, opt_emi, history = step4_optimize(
                            test_pcb, model, input_size)
    step5_report(
        test_pcb, opt_pcb,
        initial_emi, opt_emi,
        scores, causes, history
    )

    print("\n  Pipeline complete!")
    print("  Check outputs/ folder for all results")
    print("  Run: streamlit run app.py for dashboard\n")
