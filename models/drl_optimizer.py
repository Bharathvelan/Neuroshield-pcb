# drl_optimizer.py - COMPLETE FIXED VERSION

import numpy as np
import torch
import sys
import os
import json
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append('data')
sys.path.append('models')
sys.path.append('utils')

from kan_pinn import KANPINN
from graph_builder import build_pcb_graph, graph_to_feature_vector


# ─────────────────────────────────────────
# PART 1: EMI Calculator
# ─────────────────────────────────────────

def calculate_emi(pcb_dict, model, input_size):
    """Calculate EMI for a given PCB design"""
    pcb_series = pd.Series(pcb_dict)
    G  = build_pcb_graph(pcb_series)
    fv = graph_to_feature_vector(G)
    fv = fv[:input_size]
    X  = torch.FloatTensor(fv.reshape(1, -1))
    with torch.no_grad():
        emi = model(X).item()
    return float(emi)


# ─────────────────────────────────────────
# PART 2: Optimizer
# ─────────────────────────────────────────

def optimize_pcb(initial_pcb, model, input_size,
                 iterations=2000):
    """
    Optimizes PCB using physics-guided hill climbing.
    Each step tries a change guided by EMI physics rules.
    Keeps changes that reduce EMI, discards ones that dont.
    """

    print("\n" + "=" * 55)
    print("  NeuroShield-PCB: PCB Optimizer")
    print("=" * 55)

    current_pcb = initial_pcb.copy()
    initial_emi = calculate_emi(
        current_pcb, model, input_size
    )
    current_emi = initial_emi
    best_pcb    = current_pcb.copy()
    best_emi    = current_emi

    print(f"\n  Initial EMI:  {initial_emi:.2f} dBm")
    print(f"  Target:       < 40.00 dBm (CISPR 32)")
    print(f"  Iterations:   {iterations}")
    print(f"  {'-'*45}")

    LIMIT = 40.0

    # Parameter: (min, max, small_step, big_step)
    param_config = {
        'trace_width_mm':     (0.1,  2.0,   0.1,  0.3),
        'trace_length_mm':    (5.0,  100.0, 5.0,  15.0),
        'ground_distance_mm': (0.1,  2.0,   0.1,  0.3),
        'stitching_vias':     (0.0,  10.0,  1.0,  2.0),
        'decap_distance_mm':  (0.5,  15.0,  0.5,  2.0),
    }

    # Physics-guided preferred directions
    # (which direction reduces EMI for each parameter)
    preferred_direction = {
        'trace_width_mm':     +1,  # wider = less EMI
        'trace_length_mm':    -1,  # shorter = less EMI
        'ground_distance_mm': -1,  # closer ground = less EMI
        'stitching_vias':     +1,  # more vias = less EMI
        'decap_distance_mm':  -1,  # closer decap = less EMI
    }

    history = []

    for iteration in range(iterations):

        # Use big steps early, small steps later
        progress = iteration / iterations
        use_big_step = (progress < 0.5)

        # Pick parameter to adjust
        param = np.random.choice(list(param_config.keys()))
        min_val, max_val, small_step, big_step = \
            param_config[param]

        step = big_step if use_big_step else small_step

        # Mostly follow physics, sometimes explore randomly
        preferred = preferred_direction[param]
        if np.random.random() < 0.8:
            direction = preferred
        else:
            direction = -preferred

        # Apply change
        new_pcb = current_pcb.copy()
        new_val = float(new_pcb[param]) + direction * step
        new_pcb[param] = float(
            np.clip(new_val, min_val, max_val)
        )

        # Calculate new EMI
        new_emi = calculate_emi(new_pcb, model, input_size)

        # Accept if better
        if new_emi < current_emi:
            current_pcb = new_pcb.copy()
            current_emi = new_emi

            # Update best
            if new_emi < best_emi:
                best_emi = new_emi
                best_pcb = new_pcb.copy()

        # Occasionally accept worse (escape local minima)
        elif np.random.random() < 0.05 * (1 - progress):
            current_pcb = new_pcb.copy()
            current_emi = new_emi

        history.append(best_emi)

        # Print every 200 iterations
        if (iteration + 1) % 200 == 0:
            print(f"  Iter [{iteration+1:5d}/{iterations}] | "
                  f"Current: {current_emi:.2f} dBm | "
                  f"Best: {best_emi:.2f} dBm")

        # Stop early if target reached
        if best_emi < LIMIT:
            print(f"\n  Target reached at "
                  f"iteration {iteration + 1}!")
            break

    return best_pcb, best_emi, initial_emi, history


# ─────────────────────────────────────────
# PART 3: Show Results
# ─────────────────────────────────────────

def show_results(initial_pcb, optimized_pcb,
                 initial_emi, optimized_emi):
    """Prints clear before vs after comparison"""

    LIMIT       = 40.0
    improvement = initial_emi - optimized_emi

    print("\n" + "=" * 55)
    print("  OPTIMIZATION RESULTS")
    print("=" * 55)

    print(f"\n  {'Parameter':<25} {'Before':>8} {'After':>8}")
    print(f"  {'-'*43}")

    params = [
        ('trace_width_mm',     'Trace Width (mm)'),
        ('trace_length_mm',    'Trace Length (mm)'),
        ('ground_distance_mm', 'Ground Distance (mm)'),
        ('stitching_vias',     'Stitching Vias'),
        ('decap_distance_mm',  'Decap Distance (mm)'),
        ('frequency_mhz',      'Frequency (MHz)'),
    ]

    for key, label in params:
        before  = float(initial_pcb.get(key, 0))
        after   = float(optimized_pcb.get(key, 0))
        changed = ' <' if abs(before - after) > 0.01 else ''
        print(f"  {label:<25} {before:>8.1f}"
              f" {after:>8.1f}{changed}")

    print(f"\n  {'─'*43}")
    print(f"  {'EMI Before:':<25} {initial_emi:>8.2f} dBm")
    print(f"  {'EMI After:':<25} {optimized_emi:>8.2f} dBm")
    print(f"  {'Improvement:':<25} {improvement:>8.2f} dBm")
    print(f"  {'CISPR 32 Limit:':<25} {LIMIT:>8.2f} dBm")

    status_before = "FAIL" if initial_emi  > LIMIT else "PASS"
    status_after  = "PASS" if optimized_emi < LIMIT else "FAIL"

    print(f"\n  Before: {status_before}")
    print(f"  After:  {status_after}")

    if optimized_emi < LIMIT:
        print(f"\n  SUCCESS! Reduced by {improvement:.2f} dBm")
        print(f"  Board now passes CISPR 32!")
    else:
        remaining = optimized_emi - LIMIT
        print(f"\n  {remaining:.2f} dBm still over limit.")
        print(f"  But {improvement:.2f} dBm improvement achieved!")

    print("=" * 55)


# ─────────────────────────────────────────
# PART 4: Save Results
# ─────────────────────────────────────────

def save_results(optimized_pcb, initial_emi,
                 optimized_emi):
    """Save results to JSON file"""

    os.makedirs('outputs', exist_ok=True)

    # Convert numpy types to plain Python floats
    clean_pcb = {
        key: float(value)
        for key, value in optimized_pcb.items()
    }

    results = {
        'optimized_pcb':  clean_pcb,
        'initial_emi':    float(initial_emi),
        'optimized_emi':  float(optimized_emi),
        'improvement_db': float(initial_emi - optimized_emi),
        'passes_cispr32': bool(optimized_emi < 40.0)
    }

    with open('outputs/optimized_pcb.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to outputs/optimized_pcb.json")


# ─────────────────────────────────────────
# PART 5: Plot Progress
# ─────────────────────────────────────────

def plot_progress(history, initial_emi):
    """Plot EMI improvement over iterations"""

    plt.figure(figsize=(10, 5))
    plt.plot(history, color='blue',
             linewidth=1.5, label='Best EMI')
    plt.axhline(y=40.0, color='red',
                linestyle='--',
                label='CISPR 32 Limit (40 dBm)')
    plt.axhline(y=initial_emi, color='orange',
                linestyle='--',
                label=f'Initial EMI ({initial_emi:.1f} dBm)')
    plt.fill_between(
        range(len(history)),
        history, 40.0,
        where=[h > 40.0 for h in history],
        alpha=0.2, color='red', label='Violation Zone'
    )
    plt.fill_between(
        range(len(history)),
        history, 40.0,
        where=[h <= 40.0 for h in history],
        alpha=0.2, color='green', label='Safe Zone'
    )
    plt.xlabel('Iteration')
    plt.ylabel('Best EMI Found (dBm)')
    plt.title('NeuroShield-PCB: Optimization Progress')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/optimization_progress.png')
    print("Progress chart saved!")


# ─────────────────────────────────────────
# PART 6: Main
# ─────────────────────────────────────────

if __name__ == "__main__":

    # Load model
    print("Loading KAN-PINN model...")
    checkpoint = torch.load(
        'outputs/kan_pinn_model.pth',
        weights_only=False
    )
    input_size = checkpoint['input_size']
    model      = KANPINN(input_size=input_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded! Input size: {input_size}")

    # Define bad PCB
    # Frequency kept at 500MHz so optimizer
    # has realistic chance of finding solution
    bad_pcb = {
        'trace_width_mm':     0.2,
        'trace_length_mm':    90.0,
        'ground_distance_mm': 1.5,
        'stitching_vias':     1.0,
        'decap_distance_mm':  12.0,
        'frequency_mhz':      500.0
    }

    print("\nStarting with a BAD PCB design...")

    # Run optimizer
    best_pcb, best_emi, initial_emi, history = optimize_pcb(
        bad_pcb, model, input_size,
        iterations=2000
    )

    # Show results
    show_results(
        bad_pcb, best_pcb,
        initial_emi, best_emi
    )

    # Save results
    save_results(best_pcb, initial_emi, best_emi)

    # Plot progress
    plot_progress(history, initial_emi)

    print("\nAll done!")

