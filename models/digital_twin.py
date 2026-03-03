# digital_twin.py
# Phase 6: Digital Twin System
#
# Simulates real EMC lab measurements
# and uses them to continuously update
# the AI model — getting smarter over time!
#
# In production this would connect to:
# - Real EMC test chambers
# - Network analyzers
# - Spectrum analyzers
# - Automated test equipment (ATE)

import numpy as np
import torch
import torch.optim as optim
import pandas as pd
import json
import os
import sys
import copy
import matplotlib.pyplot as plt
from datetime import datetime

sys.path.append('data')
sys.path.append('models')
sys.path.append('utils')

from kan_pinn import KANPINN, physics_informed_loss
from graph_builder import (
    build_pcb_graph,
    graph_to_feature_vector
)
from real_pcb_simulator import calculate_real_emi


# ─────────────────────────────────────────
# PART 1: EMC LAB SIMULATOR
# Simulates real measurement equipment
# In production: replace with real API
# ─────────────────────────────────────────

class EMCLabSimulator:
    """
    Simulates a real EMC test chamber.

    In production this class would:
    - Connect to spectrum analyzer via GPIB
    - Trigger automated test sequence
    - Read peak EMI at each frequency
    - Return calibrated dBm measurement

    For simulation: adds realistic noise
    and systematic errors to physics model.
    """

    def __init__(self, noise_std=1.5,
                 systematic_error=0.0):
        self.noise_std        = noise_std
        self.systematic_error = systematic_error
        self.measurement_log  = []
        self.measurement_count = 0

    def measure(self, pcb_params):
        """
        Simulate a real EMC measurement.
        Returns measured EMI in dBm.
        """

        # Get physics-based true value
        true_emi = calculate_real_emi(
            trace_width_mm=
                pcb_params['trace_width_mm'],
            trace_length_mm=
                pcb_params['trace_length_mm'],
            ground_distance_mm=
                pcb_params['ground_distance_mm'],
            stitching_vias=
                int(pcb_params['stitching_vias']),
            decap_distance_mm=
                pcb_params['decap_distance_mm'],
            frequency_mhz=
                pcb_params['frequency_mhz'],
            current_ma=pcb_params.get(
                'current_ma', 10.0
            )
        )

        # Add realistic measurement noise
        noise = np.random.normal(
            0, self.noise_std
        )

        # Add systematic error
        # (instrument calibration drift)
        systematic = self.systematic_error

        # Final measurement
        measured = float(
            true_emi + noise + systematic
        )

        # Log measurement
        self.measurement_count += 1
        record = {
            'id':         self.measurement_count,
            'timestamp':  datetime.now().isoformat(),
            'params':     pcb_params.copy(),
            'true_emi':   float(true_emi),
            'measured':   measured,
            'noise':      float(noise),
            'systematic': float(systematic)
        }
        self.measurement_log.append(record)

        return measured

    def get_log(self):
        return self.measurement_log


# ─────────────────────────────────────────
# PART 2: PREDICTION ENGINE
# Uses AI model to predict EMI
# ─────────────────────────────────────────

class PredictionEngine:
    """
    Wraps the KAN-PINN model for
    easy prediction and tracking.
    """

    def __init__(self, model, input_size):
        self.model      = model
        self.input_size = input_size
        self.pred_log   = []

    def predict(self, pcb_params):
        """Predict EMI for a PCB design"""
        pcb_series = pd.Series(pcb_params)
        G  = build_pcb_graph(pcb_series)
        fv = graph_to_feature_vector(G)
        fv = fv[:self.input_size]
        X  = torch.FloatTensor(
            fv.reshape(1, -1)
        )
        with torch.no_grad():
            pred = self.model(X).item()

        self.pred_log.append(float(pred))
        return float(pred)

    def get_log(self):
        return self.pred_log


# ─────────────────────────────────────────
# PART 3: DRIFT DETECTOR
# Detects when AI becomes inaccurate
# Triggers retraining when needed
# ─────────────────────────────────────────

class DriftDetector:
    """
    Monitors prediction accuracy over time.

    Detects two types of drift:
    1. Sudden drift: one large error
       (e.g. new board type, equipment change)
    2. Gradual drift: slowly increasing error
       (e.g. component aging, process change)

    Uses CUSUM algorithm — same method used
    in industrial quality control systems!
    """

    def __init__(self,
                 warning_threshold=3.0,
                 critical_threshold=5.0,
                 window_size=10):
        self.warning_threshold  = warning_threshold
        self.critical_threshold = critical_threshold
        self.window_size        = window_size
        self.errors             = []
        self.cusum_pos          = 0.0
        self.cusum_neg          = 0.0
        self.drift_events       = []

    def update(self, predicted, measured):
        """
        Update with new prediction error.
        Returns drift status.
        """
        error     = abs(predicted - measured)
        raw_error = predicted - measured
        self.errors.append(error)

        # CUSUM update
        target  = self.warning_threshold / 2
        self.cusum_pos = max(
            0,
            self.cusum_pos + raw_error - target
        )
        self.cusum_neg = max(
            0,
            self.cusum_neg - raw_error - target
        )

        # Recent window MAE
        recent = self.errors[-self.window_size:]
        recent_mae = np.mean(recent)

        # Determine status
        if (error > self.critical_threshold or
                self.cusum_pos >
                self.critical_threshold * 3 or
                self.cusum_neg >
                self.critical_threshold * 3):
            status = 'CRITICAL'
        elif (error > self.warning_threshold or
              recent_mae > self.warning_threshold):
            status = 'WARNING'
        else:
            status = 'HEALTHY'

        if status in ['WARNING', 'CRITICAL']:
            self.drift_events.append({
                'measurement': len(self.errors),
                'error':       float(error),
                'status':      status,
                'recent_mae':  float(recent_mae)
            })

        return {
            'status':     status,
            'error':      float(error),
            'recent_mae': float(recent_mae),
            'cusum_pos':  float(self.cusum_pos),
            'cusum_neg':  float(self.cusum_neg)
        }

    def needs_retraining(self):
        """Check if model needs retraining"""
        if len(self.errors) < self.window_size:
            return False
        recent = self.errors[-self.window_size:]
        return np.mean(recent) > \
            self.warning_threshold

    def get_health_score(self):
        """
        Returns 0-100 health score.
        100 = perfect accuracy
        0   = completely inaccurate
        """
        if not self.errors:
            return 100.0
        recent = self.errors[
            -self.window_size:
        ]
        mae = np.mean(recent)
        score = max(
            0,
            100 - (mae / self.critical_threshold)
            * 100
        )
        return float(score)


# ─────────────────────────────────────────
# PART 4: AUTO-UPDATER
# Updates model when drift detected
# ─────────────────────────────────────────

class ModelAutoUpdater:
    """
    Automatically retrains the model
    when drift is detected.

    Uses online learning — updates model
    incrementally on new measurements
    without forgetting old knowledge.

    This is called 'continual learning'
    or 'lifelong learning' in AI research!
    """

    def __init__(self, model, input_size,
                 learning_rate=0.0001):
        self.model         = model
        self.input_size    = input_size
        self.optimizer     = optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=1e-6
        )
        self.update_count  = 0
        self.update_log    = []

    def update(self, pcb_params,
               measured_emi,
               y_mean=35.0, y_std=10.0):
        """
        Update model with one new measurement.
        This is online/incremental learning!
        """

        # Prepare features
        pcb_series = pd.Series(pcb_params)
        G  = build_pcb_graph(pcb_series)
        fv = graph_to_feature_vector(G)
        fv = fv[:self.input_size]
        X  = torch.FloatTensor(
            fv.reshape(1, -1)
        )

        # Normalize target
        y_norm = (measured_emi - y_mean) / y_std
        y      = torch.FloatTensor(
            [[y_norm]]
        )

        # Gradient update
        self.model.train()
        self.optimizer.zero_grad()
        pred       = self.model(X)
        loss, _    = physics_informed_loss(
            pred, y, X
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 0.5
        )
        self.optimizer.step()
        self.model.eval()

        self.update_count += 1
        self.update_log.append({
            'update':   self.update_count,
            'loss':     float(loss.item()),
            'measured': float(measured_emi)
        })

        return float(loss.item())


# ─────────────────────────────────────────
# PART 5: DIGITAL TWIN ENGINE
# Orchestrates everything together
# ─────────────────────────────────────────

class DigitalTwinEngine:
    """
    Main digital twin orchestrator.

    Connects:
    - EMC Lab (real or simulated measurements)
    - AI Prediction Engine
    - Drift Detector
    - Auto Updater

    Runs continuous predict → measure →
    compare → update loop.
    """

    def __init__(self, model, input_size,
                 noise_std=1.5):
        self.input_size = input_size

        # Components
        self.lab       = EMCLabSimulator(
            noise_std=noise_std
        )
        self.predictor = PredictionEngine(
            model, input_size
        )
        self.detector  = DriftDetector(
            warning_threshold=3.0,
            critical_threshold=5.0
        )
        self.updater   = ModelAutoUpdater(
            model, input_size
        )

        # Results tracking
        self.results   = []
        self.cycle     = 0

    def run_cycle(self, pcb_params):
        """
        One digital twin cycle:
        1. Predict EMI with AI
        2. Measure EMI in lab
        3. Compare prediction vs measurement
        4. Detect drift
        5. Update model if needed
        """

        self.cycle += 1

        # Step 1: AI prediction
        predicted = self.predictor.predict(
            pcb_params
        )

        # Step 2: Lab measurement
        measured = self.lab.measure(pcb_params)

        # Step 3: Compare
        error = abs(predicted - measured)

        # Step 4: Drift detection
        drift_status = self.detector.update(
            predicted, measured
        )

        # Step 5: Auto-update if needed
        updated = False
        update_loss = None

        if self.detector.needs_retraining():
            update_loss = self.updater.update(
                pcb_params, measured
            )
            updated = True

        # Health score
        health = self.detector.get_health_score()

        result = {
            'cycle':      self.cycle,
            'predicted':  round(predicted, 2),
            'measured':   round(measured, 2),
            'error':      round(error, 2),
            'status':     drift_status['status'],
            'health':     round(health, 1),
            'updated':    updated,
            'update_loss': round(
                update_loss, 4
            ) if update_loss else None
        }

        self.results.append(result)

        return result

    def run_simulation(self,
                       num_cycles=50,
                       vary_params=True):
        """
        Run full digital twin simulation.
        Simulates 50 real measurements
        coming in over time.
        """

        print("\n" + "=" * 60)
        print(
            "  NeuroShield-PCB: Digital Twin"
        )
        print("=" * 60)
        print(
            f"\n  Cycles:     {num_cycles}"
        )
        print(
            f"  Noise:      "
            f"±{self.lab.noise_std} dBm"
        )
        print(
            f"  Auto-update: Enabled"
        )
        print(f"  {'-'*50}")

        # Base PCB design
        base_pcb = {
            'trace_width_mm':     0.5,
            'trace_length_mm':    50.0,
            'ground_distance_mm': 0.5,
            'stitching_vias':     3.0,
            'decap_distance_mm':  5.0,
            'frequency_mhz':      500.0,
            'current_ma':         10.0
        }

        for cycle in range(num_cycles):

            # Vary parameters slightly
            # to simulate different boards
            if vary_params:
                pcb = {
                    'trace_width_mm':
                        float(np.clip(
                            base_pcb[
                                'trace_width_mm'
                            ] +
                            np.random.uniform(
                                -0.2, 0.2
                            ),
                            0.1, 2.0
                        )),
                    'trace_length_mm':
                        float(np.clip(
                            base_pcb[
                                'trace_length_mm'
                            ] +
                            np.random.uniform(
                                -10, 10
                            ),
                            5.0, 100.0
                        )),
                    'ground_distance_mm':
                        float(base_pcb[
                            'ground_distance_mm'
                        ]),
                    'stitching_vias':
                        float(base_pcb[
                            'stitching_vias'
                        ]),
                    'decap_distance_mm':
                        float(np.clip(
                            base_pcb[
                                'decap_distance_mm'
                            ] +
                            np.random.uniform(
                                -1, 1
                            ),
                            0.5, 15.0
                        )),
                    'frequency_mhz':
                        float(np.clip(
                            base_pcb[
                                'frequency_mhz'
                            ] +
                            np.random.uniform(
                                -50, 50
                            ),
                            30.0, 1000.0
                        )),
                    'current_ma':
                        float(base_pcb[
                            'current_ma'
                        ])
                }
            else:
                pcb = base_pcb.copy()

            # Run one cycle
            result = self.run_cycle(pcb)

            # Print progress
            updated_str = (
                " [AUTO-UPDATED]"
                if result['updated']
                else ""
            )
            print(
                f"  Cycle {result['cycle']:3d} | "
                f"Pred: {result['predicted']:6.2f} | "
                f"Meas: {result['measured']:6.2f} | "
                f"Err: {result['error']:5.2f} | "
                f"{result['status']:<8}"
                f"{updated_str}"
            )

        return self.results

    def get_summary(self):
        """Summary of digital twin session"""
        if not self.results:
            return {}

        errors  = [r['error'] for r in self.results]
        updates = sum(
            1 for r in self.results
            if r['updated']
        )
        healthy = sum(
            1 for r in self.results
            if r['status'] == 'HEALTHY'
        )
        warning = sum(
            1 for r in self.results
            if r['status'] == 'WARNING'
        )
        critical = sum(
            1 for r in self.results
            if r['status'] == 'CRITICAL'
        )

        return {
            'total_cycles':    len(self.results),
            'avg_error':       round(
                float(np.mean(errors)), 3
            ),
            'max_error':       round(
                float(np.max(errors)), 3
            ),
            'auto_updates':    updates,
            'healthy_cycles':  healthy,
            'warning_cycles':  warning,
            'critical_cycles': critical,
            'final_health':    round(
                self.detector.get_health_score(),
                1
            )
        }

    def plot_results(self):
        """Plot digital twin results"""

        cycles    = [r['cycle']
                     for r in self.results]
        predicted = [r['predicted']
                     for r in self.results]
        measured  = [r['measured']
                     for r in self.results]
        errors    = [r['error']
                     for r in self.results]
        health    = [r['health']
                     for r in self.results]

        fig, axes = plt.subplots(
            2, 2, figsize=(14, 10)
        )

        # Plot 1: Prediction vs Measurement
        axes[0, 0].plot(
            cycles, predicted,
            color='blue', linewidth=1.5,
            label='AI Prediction', alpha=0.8
        )
        axes[0, 0].plot(
            cycles, measured,
            color='red', linewidth=1.5,
            linestyle='--',
            label='Lab Measurement', alpha=0.8
        )
        axes[0, 0].axhline(
            y=40.0, color='orange',
            linestyle=':', linewidth=1.5,
            label='CISPR 32 Limit'
        )
        axes[0, 0].set_xlabel('Measurement Cycle')
        axes[0, 0].set_ylabel('EMI (dBm)')
        axes[0, 0].set_title(
            'Prediction vs Measurement',
            fontweight='bold'
        )
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Prediction Error
        colors_e = [
            'red'    if r['status'] == 'CRITICAL'
            else 'orange'
            if r['status'] == 'WARNING'
            else 'green'
            for r in self.results
        ]
        axes[0, 1].bar(
            cycles, errors,
            color=colors_e, alpha=0.7
        )
        axes[0, 1].axhline(
            y=3.0, color='orange',
            linestyle='--',
            label='Warning (3 dBm)'
        )
        axes[0, 1].axhline(
            y=5.0, color='red',
            linestyle='--',
            label='Critical (5 dBm)'
        )
        axes[0, 1].set_xlabel('Measurement Cycle')
        axes[0, 1].set_ylabel('Error (dBm)')
        axes[0, 1].set_title(
            'Prediction Error per Cycle',
            fontweight='bold'
        )
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Model Health Score
        axes[1, 0].plot(
            cycles, health,
            color='green', linewidth=2,
            label='Health Score'
        )
        axes[1, 0].axhline(
            y=70, color='orange',
            linestyle='--',
            label='Warning Threshold'
        )
        axes[1, 0].axhline(
            y=40, color='red',
            linestyle='--',
            label='Critical Threshold'
        )
        axes[1, 0].fill_between(
            cycles, health, 70,
            where=[h < 70 for h in health],
            alpha=0.2, color='orange'
        )
        axes[1, 0].set_xlabel('Measurement Cycle')
        axes[1, 0].set_ylabel('Health Score')
        axes[1, 0].set_ylim(0, 105)
        axes[1, 0].set_title(
            'Model Health Score Over Time',
            fontweight='bold'
        )
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Auto-updates timeline
        update_cycles = [
            r['cycle'] for r in self.results
            if r['updated']
        ]
        update_errors = [
            r['error'] for r in self.results
            if r['updated']
        ]
        axes[1, 1].scatter(
            cycles, errors,
            c=colors_e, alpha=0.5,
            label='All measurements'
        )
        if update_cycles:
            axes[1, 1].scatter(
                update_cycles,
                update_errors,
                color='purple',
                s=100, zorder=5,
                marker='*',
                label='Auto-update triggered'
            )
        axes[1, 1].set_xlabel('Measurement Cycle')
        axes[1, 1].set_ylabel('Error (dBm)')
        axes[1, 1].set_title(
            'Auto-Update Events',
            fontweight='bold'
        )
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle(
            'NeuroShield-PCB: Digital Twin Results',
            fontsize=14,
            fontweight='bold'
        )
        plt.tight_layout()
        plt.savefig(
            'outputs/digital_twin_results.png',
            dpi=150,
            bbox_inches='tight'
        )
        print(
            "  Chart saved to "
            "outputs/digital_twin_results.png"
        )
        return fig


# ─────────────────────────────────────────
# MAIN: RUN DIGITAL TWIN
# ─────────────────────────────────────────

if __name__ == "__main__":

    print("Loading KAN-PINN model...")
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
    print(f"Model loaded! Input size: {input_size}")

    # Create digital twin
    twin = DigitalTwinEngine(
        model, input_size,
        noise_std=1.5
    )

    # Run 50 measurement cycles
    results = twin.run_simulation(
        num_cycles=50,
        vary_params=True
    )

    # Summary
    summary = twin.get_summary()

    print("\n" + "=" * 60)
    print("  DIGITAL TWIN SUMMARY")
    print("=" * 60)
    print(
        f"  Total Cycles:   "
        f"{summary['total_cycles']}"
    )
    print(
        f"  Avg Error:      "
        f"{summary['avg_error']} dBm"
    )
    print(
        f"  Max Error:      "
        f"{summary['max_error']} dBm"
    )
    print(
        f"  Auto Updates:   "
        f"{summary['auto_updates']}"
    )
    print(
        f"  Healthy Cycles: "
        f"{summary['healthy_cycles']}"
    )
    print(
        f"  Warning Cycles: "
        f"{summary['warning_cycles']}"
    )
    print(
        f"  Final Health:   "
        f"{summary['final_health']}%"
    )

    # Save results
    os.makedirs('outputs', exist_ok=True)
    with open(
        'outputs/digital_twin_results.json', 'w'
    ) as f:
        json.dump({
            'summary': summary,
            'cycles':  results
        }, f, indent=2)
    print(
        "\n  Results saved to "
        "outputs/digital_twin_results.json"
    )

    # Plot
    twin.plot_results()
    print("=" * 60)
