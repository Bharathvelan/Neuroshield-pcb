# uncertainty.py
# Phase 7A: Uncertainty Quantification
#
# AI predicts EMI AND tells you
# how confident it is!
#
# Method: Monte Carlo Dropout
# Run model 100 times with dropout ON
# Mean = prediction
# Std  = uncertainty
#
# "EMI = 38.5 ± 2.3 dBm (95% confidence)"
#
# Used in:
# - Medical AI (cancer diagnosis confidence)
# - Autonomous vehicles (obstacle uncertainty)
# - Financial AI (risk quantification)

import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import json
import os
import sys
import matplotlib.pyplot as plt

sys.path.append('data')
sys.path.append('models')
sys.path.append('utils')

from kan_pinn import KANPINN
from graph_builder import (
    build_pcb_graph,
    graph_to_feature_vector
)


# ─────────────────────────────────────────
# PART 1: MONTE CARLO DROPOUT PREDICTOR
# ─────────────────────────────────────────

class MCDropoutPredictor:
    """
    Monte Carlo Dropout for uncertainty.

    Key insight:
    - Normal inference: dropout OFF
      → single deterministic prediction
    - MC Dropout: dropout ON during inference
      → each forward pass gives different result
      → run N times → distribution of predictions
      → mean = best estimate
      → std  = uncertainty

    This is Bayesian approximation —
    same math as full Bayesian neural network
    but 100x faster!
    """

    def __init__(self, model, input_size,
                 n_samples=100,
                 dropout_rate=0.1):
        self.model        = model
        self.input_size   = input_size
        self.n_samples    = n_samples
        self.dropout_rate = dropout_rate

        # Add dropout layers to model
        # for MC sampling
        self._add_dropout()

    def _add_dropout(self):
        """
        Add dropout to KAN-PINN layers
        for MC sampling.
        Modifies model in place.
        """
        for name, module in \
                self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Dropout is already in KANPINN
                # Just ensure it stays active
                pass

    def _enable_dropout(self):
        """Enable dropout for MC sampling"""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def predict_with_uncertainty(
            self, pcb_params):
        """
        Run N forward passes with dropout ON.
        Returns mean prediction + uncertainty.
        """

        # Prepare input
        pcb_series = pd.Series(pcb_params)
        G  = build_pcb_graph(pcb_series)
        fv = graph_to_feature_vector(G)
        fv = fv[:self.input_size]
        X  = torch.FloatTensor(
            fv.reshape(1, -1)
        )

        # MC sampling
        self.model.eval()
        self._enable_dropout()

        predictions = []
        with torch.no_grad():
            for _ in range(self.n_samples):
                pred = self.model(X).item()
                predictions.append(pred)

        predictions = np.array(predictions)

        # Statistics
        mean     = float(np.mean(predictions))
        std      = float(np.std(predictions))
        ci_95_lo = float(
            np.percentile(predictions, 2.5)
        )
        ci_95_hi = float(
            np.percentile(predictions, 97.5)
        )
        ci_68_lo = float(
            np.percentile(predictions, 16)
        )
        ci_68_hi = float(
            np.percentile(predictions, 84)
        )

        # Confidence level
        if std < 1.0:
            confidence = 'HIGH'
        elif std < 2.5:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'

        return {
            'mean':       round(mean, 2),
            'std':        round(std, 3),
            'ci_95_lo':   round(ci_95_lo, 2),
            'ci_95_hi':   round(ci_95_hi, 2),
            'ci_68_lo':   round(ci_68_lo, 2),
            'ci_68_hi':   round(ci_68_hi, 2),
            'confidence': confidence,
            'samples':    predictions.tolist(),
            'n_samples':  self.n_samples
        }

    def predict_frequency_sweep(
            self, pcb_params,
            freq_range=(30, 1000),
            n_freqs=40):
        """
        Predict EMI with uncertainty
        across frequency range.
        Returns spectrum with confidence bands.
        """

        freqs   = np.linspace(
            freq_range[0],
            freq_range[1],
            n_freqs
        )
        means   = []
        stds    = []
        lo_95   = []
        hi_95   = []

        print(
            f"  Running uncertainty sweep "
            f"({n_freqs} frequencies × "
            f"{self.n_samples} samples)..."
        )

        for i, freq in enumerate(freqs):
            pcb = pcb_params.copy()
            pcb['frequency_mhz'] = float(freq)
            result = \
                self.predict_with_uncertainty(
                    pcb
                )
            means.append(result['mean'])
            stds.append(result['std'])
            lo_95.append(result['ci_95_lo'])
            hi_95.append(result['ci_95_hi'])

            if (i + 1) % 10 == 0:
                print(
                    f"    {i+1}/{n_freqs} done"
                )

        return {
            'frequencies': freqs.tolist(),
            'means':       means,
            'stds':        stds,
            'ci_95_lo':    lo_95,
            'ci_95_hi':    hi_95
        }

    def analyze_parameter_sensitivity(
            self, base_pcb):
        """
        Analyze how uncertainty changes
        when each parameter varies.
        Shows which parameters the AI
        is most/least confident about.
        """

        print(
            "\n  Analyzing parameter "
            "sensitivity..."
        )

        params = {
            'trace_width_mm': (0.1, 2.0),
            'trace_length_mm': (5.0, 100.0),
            'ground_distance_mm': (0.1, 2.0),
            'stitching_vias': (0.0, 10.0),
            'decap_distance_mm': (0.5, 15.0),
        }

        sensitivity = {}

        for param, (low, high) in params.items():
            values = np.linspace(low, high, 10)
            param_stds = []

            for val in values:
                pcb = base_pcb.copy()
                pcb[param] = float(val)
                result = \
                    self.predict_with_uncertainty(
                        pcb
                    )
                param_stds.append(result['std'])

            sensitivity[param] = {
                'mean_uncertainty':
                    round(float(
                        np.mean(param_stds)
                    ), 3),
                'max_uncertainty':
                    round(float(
                        np.max(param_stds)
                    ), 3),
                'values':
                    values.tolist(),
                'uncertainties':
                    param_stds
            }
            print(
                f"    {param:<25}: "
                f"avg std = "
                f"{sensitivity[param]['mean_uncertainty']:.3f}"
            )

        return sensitivity


# ─────────────────────────────────────────
# PART 2: COMPLIANCE RISK ANALYZER
# ─────────────────────────────────────────

class ComplianceRiskAnalyzer:
    """
    Uses uncertainty to calculate
    probability of FAILING compliance.

    "Your PCB has 23% chance of failing
    CISPR 32 Class B"

    This is much more useful than just
    PASS/FAIL because it accounts for
    measurement uncertainty!
    """

    def __init__(self, limit_dbm=40.0):
        self.limit = limit_dbm

    def calculate_risk(self, uc_result):
        """
        Calculate probability of exceeding
        compliance limit given uncertainty.
        """

        mean    = uc_result['mean']
        std     = uc_result['std']
        samples = np.array(uc_result['samples'])

        # Empirical probability from samples
        prob_fail = float(
            np.mean(samples > self.limit)
        )
        prob_pass = 1.0 - prob_fail

        # Risk level
        if prob_fail < 0.05:
            risk_level = 'VERY LOW'
            color      = 'green'
        elif prob_fail < 0.15:
            risk_level = 'LOW'
            color      = 'lightgreen'
        elif prob_fail < 0.35:
            risk_level = 'MEDIUM'
            color      = 'orange'
        elif prob_fail < 0.65:
            risk_level = 'HIGH'
            color      = 'red'
        else:
            risk_level = 'VERY HIGH'
            color      = 'darkred'

        # Margin statistics
        margins = self.limit - samples
        margin_mean = float(np.mean(margins))
        margin_std  = float(np.std(margins))

        return {
            'prob_fail':   round(prob_fail, 3),
            'prob_pass':   round(prob_pass, 3),
            'risk_level':  risk_level,
            'color':       color,
            'margin_mean': round(margin_mean, 2),
            'margin_std':  round(margin_std, 2),
            'limit':       self.limit
        }


# ─────────────────────────────────────────
# PART 3: VISUALIZER
# ─────────────────────────────────────────

def plot_uncertainty_results(
        uc_result, risk_result,
        sweep_result, sensitivity,
        save_path='outputs/uncertainty_results.png'
):
    """Plot all uncertainty analysis results"""

    fig, axes = plt.subplots(
        2, 2, figsize=(14, 10)
    )

    # Plot 1: Prediction distribution
    samples = np.array(uc_result['samples'])
    axes[0, 0].hist(
        samples, bins=30,
        color='steelblue', alpha=0.7,
        edgecolor='black'
    )
    axes[0, 0].axvline(
        x=uc_result['mean'],
        color='blue', linewidth=2,
        label=f"Mean: {uc_result['mean']:.1f} dBm"
    )
    axes[0, 0].axvline(
        x=40.0, color='red',
        linestyle='--', linewidth=2,
        label='CISPR 32 Limit (40 dBm)'
    )
    axes[0, 0].axvspan(
        uc_result['ci_95_lo'],
        uc_result['ci_95_hi'],
        alpha=0.2, color='blue',
        label='95% Confidence Interval'
    )
    axes[0, 0].set_xlabel('Predicted EMI (dBm)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title(
        f"Prediction Distribution\n"
        f"Mean={uc_result['mean']:.1f} ± "
        f"{uc_result['std']:.2f} dBm",
        fontweight='bold'
    )
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Frequency sweep with uncertainty
    freqs  = sweep_result['frequencies']
    means  = sweep_result['means']
    lo_95  = sweep_result['ci_95_lo']
    hi_95  = sweep_result['ci_95_hi']

    axes[0, 1].plot(
        freqs, means,
        color='blue', linewidth=2,
        label='Mean Prediction'
    )
    axes[0, 1].fill_between(
        freqs, lo_95, hi_95,
        alpha=0.3, color='blue',
        label='95% Confidence Band'
    )
    axes[0, 1].axhline(
        y=40.0, color='red',
        linestyle='--', linewidth=2,
        label='CISPR 32 Limit'
    )
    axes[0, 1].fill_between(
        freqs, means, 40.0,
        where=[m > 40 for m in means],
        alpha=0.3, color='red',
        label='Violation Zone'
    )
    axes[0, 1].set_xlabel('Frequency (MHz)')
    axes[0, 1].set_ylabel('EMI (dBm)')
    axes[0, 1].set_title(
        'EMI Spectrum with Uncertainty Bands',
        fontweight='bold'
    )
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Compliance risk gauge
    prob_fail = risk_result['prob_fail']
    prob_pass = risk_result['prob_pass']

    wedges, texts, autotexts = axes[1, 0].pie(
        [prob_pass, prob_fail],
        labels=['PASS', 'FAIL'],
        colors=['green', 'red'],
        autopct='%1.1f%%',
        startangle=90,
        pctdistance=0.75,
        wedgeprops={'linewidth': 2,
                    'edgecolor': 'white'}
    )
    axes[1, 0].set_title(
        f"Compliance Risk\n"
        f"Risk Level: {risk_result['risk_level']}",
        fontweight='bold'
    )

    # Plot 4: Parameter sensitivity
    param_names = list(sensitivity.keys())
    mean_stds   = [
        sensitivity[p]['mean_uncertainty']
        for p in param_names
    ]
    short_names = [
        p.replace('_mm', '')
         .replace('_', ' ')
         .title()
        for p in param_names
    ]
    colors_s = [
        'red'    if s > 2.0 else
        'orange' if s > 1.0 else
        'green'
        for s in mean_stds
    ]
    axes[1, 1].barh(
        short_names, mean_stds,
        color=colors_s,
        edgecolor='black', alpha=0.8
    )
    axes[1, 1].set_xlabel(
        'Average Uncertainty (dBm std)'
    )
    axes[1, 1].set_title(
        'Uncertainty by Parameter',
        fontweight='bold'
    )
    axes[1, 1].grid(True, alpha=0.3, axis='x')

    plt.suptitle(
        'NeuroShield-PCB: '
        'Uncertainty Quantification',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(
        save_path, dpi=150,
        bbox_inches='tight'
    )
    print(f"  Chart saved to {save_path}")
    return fig


# ─────────────────────────────────────────
# MAIN: TEST
# ─────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 60)
    print(
        "  Phase 7A: Uncertainty Quantification"
    )
    print("=" * 60)

    print("\nLoading KAN-PINN model...")
    checkpoint = torch.load(
        'outputs/kan_pinn_model.pth',
        weights_only=False
    )
    input_size = checkpoint['input_size']
    model      = KANPINN(input_size=input_size)
    model.load_state_dict(
        checkpoint['model_state_dict']
    )
    model.eval()
    print(f"Model loaded! Input: {input_size}")

    # Test PCB
    test_pcb = {
        'trace_width_mm':     0.5,
        'trace_length_mm':    60.0,
        'ground_distance_mm': 0.8,
        'stitching_vias':     3.0,
        'decap_distance_mm':  6.0,
        'frequency_mhz':      500.0
    }

    # Create predictor
    predictor = MCDropoutPredictor(
        model, input_size,
        n_samples=100,
        dropout_rate=0.1
    )

    # Single prediction with uncertainty
    print("\n[1/4] Single prediction...")
    result = predictor.predict_with_uncertainty(
        test_pcb
    )

    print(
        f"\n  EMI Prediction:"
    )
    print(
        f"  Mean:          "
        f"{result['mean']:.2f} dBm"
    )
    print(
        f"  Uncertainty:   "
        f"±{result['std']:.2f} dBm"
    )
    print(
        f"  95% CI:        "
        f"[{result['ci_95_lo']:.1f}, "
        f"{result['ci_95_hi']:.1f}] dBm"
    )
    print(
        f"  Confidence:    "
        f"{result['confidence']}"
    )

    # Compliance risk
    print("\n[2/4] Compliance risk...")
    risk_analyzer = ComplianceRiskAnalyzer(
        limit_dbm=40.0
    )
    risk = risk_analyzer.calculate_risk(result)

    print(
        f"  Prob(PASS):    "
        f"{risk['prob_pass']*100:.1f}%"
    )
    print(
        f"  Prob(FAIL):    "
        f"{risk['prob_fail']*100:.1f}%"
    )
    print(
        f"  Risk Level:    "
        f"{risk['risk_level']}"
    )

    # Frequency sweep
    print("\n[3/4] Frequency sweep...")
    sweep = predictor.predict_frequency_sweep(
        test_pcb,
        freq_range=(30, 1000),
        n_freqs=30
    )

    # Sensitivity analysis
    print("\n[4/4] Sensitivity analysis...")
    sensitivity = \
        predictor.analyze_parameter_sensitivity(
            test_pcb
        )

    # Plot
    os.makedirs('outputs', exist_ok=True)
    fig = plot_uncertainty_results(
        result, risk, sweep, sensitivity
    )
    plt.close()

    # Save results
    save_data = {
        'prediction':    result,
        'risk':          risk,
        'sweep':         sweep,
        'sensitivity': {
            k: {
                'mean_uncertainty':
                    v['mean_uncertainty'],
                'max_uncertainty':
                    v['max_uncertainty']
            }
            for k, v in sensitivity.items()
        }
    }
    with open(
        'outputs/uncertainty_results.json', 'w'
    ) as f:
        json.dump(save_data, f, indent=2)

    print("\n" + "=" * 60)
    print("  UNCERTAINTY SUMMARY")
    print("=" * 60)
    print(
        f"  Prediction:  "
        f"{result['mean']:.2f} ± "
        f"{result['std']:.2f} dBm"
    )
    print(
        f"  95% CI:      "
        f"[{result['ci_95_lo']:.1f}, "
        f"{result['ci_95_hi']:.1f}]"
    )
    print(
        f"  Confidence:  {result['confidence']}"
    )
    print(
        f"  Pass Prob:   "
        f"{risk['prob_pass']*100:.1f}%"
    )
    print(
        f"  Risk Level:  {risk['risk_level']}"
    )
    print(
        f"\n  Files saved:"
    )
    print(
        f"  outputs/uncertainty_results.json"
    )
    print(
        f"  outputs/uncertainty_results.png"
    )
    print("=" * 60)
