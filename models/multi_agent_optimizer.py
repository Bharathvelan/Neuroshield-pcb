# multi_agent_optimizer.py
# Phase 3: Multi-Agent DRL System
# 4 specialized agents working as a team

import numpy as np
import torch
import pandas as pd
import sys
import os

sys.path.append('data')
sys.path.append('models')
sys.path.append('utils')

from kan_pinn import KANPINN
from graph_builder import (
    build_pcb_graph,
    graph_to_feature_vector
)


# ─────────────────────────────────────────
# HELPER: Calculate EMI
# ─────────────────────────────────────────

def get_emi(pcb_dict, model, input_size):
    pcb_series = pd.Series(pcb_dict)
    G  = build_pcb_graph(pcb_series)
    fv = graph_to_feature_vector(G)
    fv = fv[:input_size]
    X  = torch.FloatTensor(fv.reshape(1, -1))
    with torch.no_grad():
        emi = model(X).item()
    return float(emi)


# ─────────────────────────────────────────
# BASE AGENT
# ─────────────────────────────────────────

class BaseAgent:
    def __init__(self, name, model, input_size):
        self.name       = name
        self.model      = model
        self.input_size = input_size
        self.best_emi   = float('inf')
        self.best_pcb   = None

    def get_action(self, pcb, iteration,
                   max_iter):
        raise NotImplementedError

    def evaluate(self, pcb):
        return get_emi(
            pcb, self.model, self.input_size
        )

    def update_best(self, pcb, emi):
        if emi < self.best_emi:
            self.best_emi = emi
            self.best_pcb = pcb.copy()
            return True
        return False


# ─────────────────────────────────────────
# AGENT A: PDN SPECIALIST
# Controls: trace_width, decap_distance
# ─────────────────────────────────────────

class PDNAgent(BaseAgent):
    def __init__(self, model, input_size):
        super().__init__(
            'PDN Specialist', model, input_size
        )

    def get_action(self, pcb, iteration,
                   max_iter):
        new_pcb  = pcb.copy()
        progress = iteration / max_iter

        if progress < 0.4:
            step_w = 0.2
            step_d = 1.0
        elif progress < 0.7:
            step_w = 0.1
            step_d = 0.5
        else:
            step_w = 0.05
            step_d = 0.25

        # Widen trace
        new_pcb['trace_width_mm'] = float(
            np.clip(
                new_pcb['trace_width_mm'] + step_w,
                0.1, 2.0
            )
        )

        # Move decap closer
        new_pcb['decap_distance_mm'] = float(
            np.clip(
                new_pcb['decap_distance_mm'] -
                step_d,
                0.5, 15.0
            )
        )

        # Random exploration
        if np.random.random() < 0.15:
            new_pcb['trace_width_mm'] = float(
                np.clip(
                    pcb['trace_width_mm'] +
                    np.random.uniform(-0.3, 0.3),
                    0.1, 2.0
                )
            )

        return new_pcb


# ─────────────────────────────────────────
# AGENT B: SIGNAL INTEGRITY SPECIALIST
# Controls: trace_length, trace_width
# ─────────────────────────────────────────

class SignalIntegrityAgent(BaseAgent):
    def __init__(self, model, input_size):
        super().__init__(
            'SI Specialist', model, input_size
        )

    def get_action(self, pcb, iteration,
                   max_iter):
        new_pcb  = pcb.copy()
        progress = iteration / max_iter

        step_l = max(1.0, 10.0 * (1 - progress))
        step_w = max(0.05, 0.2 * (1 - progress))

        # Shorten trace
        new_pcb['trace_length_mm'] = float(
            np.clip(
                new_pcb['trace_length_mm'] - step_l,
                5.0, 100.0
            )
        )

        # Target 50 ohm impedance
        target_width = 0.5
        current      = new_pcb['trace_width_mm']
        if current < target_width:
            new_pcb['trace_width_mm'] = float(
                np.clip(
                    current + step_w,
                    0.1, 2.0
                )
            )

        # Random exploration
        if np.random.random() < 0.2:
            new_pcb['trace_length_mm'] = float(
                np.clip(
                    pcb['trace_length_mm'] +
                    np.random.uniform(-15, 5),
                    5.0, 100.0
                )
            )

        return new_pcb


# ─────────────────────────────────────────
# AGENT C: RETURN PATH SPECIALIST
# Controls: stitching_vias, ground_distance
# ─────────────────────────────────────────

class ReturnPathAgent(BaseAgent):
    def __init__(self, model, input_size):
        super().__init__(
            'Return Path', model, input_size
        )

    def get_action(self, pcb, iteration,
                   max_iter):
        new_pcb  = pcb.copy()
        progress = iteration / max_iter

        # Add vias
        if new_pcb['stitching_vias'] < 10:
            add_vias = 2 if progress < 0.5 else 1
            new_pcb['stitching_vias'] = float(
                min(
                    10.0,
                    new_pcb['stitching_vias'] +
                    add_vias
                )
            )

        # Reduce ground distance
        step_g = max(0.05, 0.2 * (1 - progress))
        new_pcb['ground_distance_mm'] = float(
            np.clip(
                new_pcb['ground_distance_mm'] -
                step_g,
                0.1, 2.0
            )
        )

        # Random exploration
        if np.random.random() < 0.1:
            new_pcb['stitching_vias'] = float(
                np.clip(
                    pcb['stitching_vias'] +
                    np.random.randint(-1, 3),
                    0, 10
                )
            )

        return new_pcb


# ─────────────────────────────────────────
# AGENT D: DECOUPLING SPECIALIST
# Controls: decap_distance, ground_distance
# ─────────────────────────────────────────

class DecouplingAgent(BaseAgent):
    def __init__(self, model, input_size):
        super().__init__(
            'Decoupling', model, input_size
        )

    def get_action(self, pcb, iteration,
                   max_iter):
        new_pcb  = pcb.copy()
        progress = iteration / max_iter

        # Minimize decap distance
        step_d = max(0.1, 2.0 * (1 - progress))
        new_pcb['decap_distance_mm'] = float(
            np.clip(
                new_pcb['decap_distance_mm'] -
                step_d,
                0.5, 15.0
            )
        )

        # Optimize ground distance
        new_pcb['ground_distance_mm'] = float(
            np.clip(
                new_pcb['ground_distance_mm'] -
                0.05,
                0.1, 2.0
            )
        )

        # Random exploration
        if np.random.random() < 0.1:
            new_pcb['decap_distance_mm'] = float(
                np.clip(
                    np.random.uniform(0.5, 5.0),
                    0.5, 15.0
                )
            )

        return new_pcb


# ─────────────────────────────────────────
# MULTI-AGENT COORDINATOR
# ─────────────────────────────────────────

class MultiAgentCoordinator:
    """
    Coordinates all 4 specialist agents.
    Each iteration all agents propose a fix.
    Best fix is accepted.
    All agents that improved get credit.
    """

    def __init__(self, model, input_size):
        self.model      = model
        self.input_size = input_size
        self.LIMIT      = 40.0

        self.agents = [
            PDNAgent(model, input_size),
            SignalIntegrityAgent(
                model, input_size
            ),
            ReturnPathAgent(model, input_size),
            DecouplingAgent(model, input_size)
        ]

        self.history          = []
        self.agent_wins       = {
            a.name: 0 for a in self.agents
        }
        self.best_emi_overall = float('inf')
        self.best_pcb_overall = None

    def optimize(self, initial_pcb,
                 iterations=1000):

        print("\n" + "=" * 60)
        print("  NeuroShield: Multi-Agent Optimizer")
        print("=" * 60)
        print(f"\n  Agents:")
        for a in self.agents:
            print(f"    → {a.name}")

        current_pcb = initial_pcb.copy()
        current_emi = get_emi(
            current_pcb, self.model,
            self.input_size
        )

        self.best_emi_overall = current_emi
        self.best_pcb_overall = current_pcb.copy()

        print(f"\n  Initial EMI: {current_emi:.2f} dBm")
        print(f"  Target:      < {self.LIMIT} dBm")
        print(f"  Iterations:  {iterations}")
        print(f"  {'-'*50}")

        for iteration in range(iterations):

            proposals = []
            for agent in self.agents:
                proposed_pcb = agent.get_action(
                    current_pcb,
                    iteration,
                    iterations
                )
                proposed_emi = get_emi(
                    proposed_pcb,
                    self.model,
                    self.input_size
                )
                proposals.append((
                    agent,
                    proposed_pcb,
                    proposed_emi
                ))

            # Give credit to ALL agents
            # that found an improvement
            for agent, prop_pcb, prop_emi \
                    in proposals:
                if prop_emi < current_emi:
                    self.agent_wins[
                        agent.name
                    ] += 1

            # Accept best proposal
            best_agent, best_pcb, best_emi = \
                min(
                    proposals,
                    key=lambda x: x[2]
                )

            if best_emi < current_emi:
                current_pcb = best_pcb.copy()
                current_emi = best_emi

                if best_emi < self.best_emi_overall:
                    self.best_emi_overall = best_emi
                    self.best_pcb_overall = \
                        best_pcb.copy()

            # Occasionally accept worse
            elif np.random.random() < 0.03:
                current_pcb = best_pcb.copy()
                current_emi = best_emi

            self.history.append(
                self.best_emi_overall
            )

            if (iteration + 1) % 200 == 0:
                print(
                    f"  Iter [{iteration+1:5d}/"
                    f"{iterations}] | "
                    f"Current: {current_emi:.2f} | "
                    f"Best: "
                    f"{self.best_emi_overall:.2f} dBm"
                )

            if self.best_emi_overall < self.LIMIT:
                print(
                    f"\n  Target reached at "
                    f"iteration {iteration+1}!"
                )
                break

        return (
            self.best_pcb_overall,
            self.best_emi_overall,
            self.history
        )

    def get_agent_report(self):
        total = sum(self.agent_wins.values())
        report = {}
        for name, wins in self.agent_wins.items():
            pct = (wins / total * 100) \
                if total > 0 else 0
            report[name] = {
                'wins':         wins,
                'contribution': f"{pct:.1f}%"
            }
        return report


# ─────────────────────────────────────────
# MAIN: TEST
# ─────────────────────────────────────────

if __name__ == "__main__":

    import json
    import matplotlib.pyplot as plt

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

    # Bad PCB
    bad_pcb = {
        'trace_width_mm':     0.2,
        'trace_length_mm':    90.0,
        'ground_distance_mm': 1.5,
        'stitching_vias':     1.0,
        'decap_distance_mm':  12.0,
        'frequency_mhz':      500.0
    }

    init_emi = get_emi(bad_pcb, model, input_size)
    print(f"\nInitial EMI: {init_emi:.2f} dBm")

    # Run optimizer
    coordinator = MultiAgentCoordinator(
        model, input_size
    )
    best_pcb, best_emi, history = \
        coordinator.optimize(
            bad_pcb, iterations=1000
        )

    improvement = init_emi - best_emi
    LIMIT       = 40.0

    # Print results
    print("\n" + "=" * 60)
    print("  MULTI-AGENT RESULTS")
    print("=" * 60)

    print(f"\n  {'Parameter':<25}"
          f" {'Before':>8} {'After':>8}")
    print(f"  {'-'*43}")

    params = [
        ('trace_width_mm',     'Trace Width'),
        ('trace_length_mm',    'Trace Length'),
        ('ground_distance_mm', 'Ground Dist'),
        ('stitching_vias',     'Vias'),
        ('decap_distance_mm',  'Decap Dist'),
        ('frequency_mhz',      'Frequency'),
    ]

    for key, label in params:
        before  = float(bad_pcb.get(key, 0))
        after   = float(best_pcb.get(key, 0))
        changed = ' <' \
            if abs(before - after) > 0.01 else ''
        print(
            f"  {label:<25}"
            f" {before:>8.1f}"
            f" {after:>8.1f}{changed}"
        )

    print(f"\n  EMI Before:  {init_emi:.2f} dBm")
    print(f"  EMI After:   {best_emi:.2f} dBm")
    print(f"  Improvement: {improvement:.2f} dBm")
    print(
        f"  Status: "
        f"{'PASS' if best_emi < LIMIT else 'FAIL'}"
    )

    # Agent report
    print("\n  Agent Contributions:")
    print(f"  {'-'*40}")
    report = coordinator.get_agent_report()
    for name, data in report.items():
        print(
            f"  {name:<20}"
            f" {data['wins']:>4} wins"
            f" ({data['contribution']})"
        )

    # Save results
    os.makedirs('outputs', exist_ok=True)
    clean = {
        k: float(v)
        for k, v in best_pcb.items()
    }
    results = {
        'initial_emi':    float(init_emi),
        'optimized_emi':  float(best_emi),
        'improvement_db': float(improvement),
        'passes_cispr32': bool(best_emi < LIMIT),
        'optimized_pcb':  clean,
        'agent_report':   report
    }
    with open(
        'outputs/multi_agent_results.json', 'w'
    ) as f:
        json.dump(results, f, indent=2)
    print("\n  Results saved!")

    # ── Plot ──
    fig, axes = plt.subplots(
        1, 2, figsize=(14, 5)
    )

    # Progress chart
    axes[0].plot(
        history, color='blue',
        linewidth=1.5, label='Best EMI'
    )
    axes[0].axhline(
        y=LIMIT, color='red',
        linestyle='--', linewidth=2,
        label=f'CISPR 32 ({LIMIT} dBm)'
    )
    axes[0].axhline(
        y=init_emi, color='orange',
        linestyle=':', linewidth=1.5,
        label=f'Initial ({init_emi:.1f} dBm)'
    )
    axes[0].fill_between(
        range(len(history)),
        history, LIMIT,
        where=[h > LIMIT for h in history],
        alpha=0.2, color='red'
    )
    axes[0].fill_between(
        range(len(history)),
        history, LIMIT,
        where=[h <= LIMIT for h in history],
        alpha=0.2, color='green'
    )
    axes[0].set_xlabel('Iteration')
    axes[0].set_ylabel('Best EMI (dBm)')
    axes[0].set_title(
        'Multi-Agent Optimization Progress'
    )
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Pie chart — short names, no overlap
    short_names = ['PDN', 'SI', 'Return', 'Decap']
    wins = [
        report[n]['wins']
        for n in report.keys()
    ]

    if sum(wins) > 0:
        axes[1].pie(
            wins,
            labels=short_names,
            autopct='%1.1f%%',
            startangle=90,
            pctdistance=0.75,
            labeldistance=1.15,
            colors=[
                '#ff6b6b', '#4ecdc4',
                '#45b7d1', '#96ceb4'
            ]
        )
        axes[1].set_title(
            'Agent Contribution to Improvement'
        )
    else:
        axes[1].text(
            0.5, 0.5,
            'No improvements found\n'
            'Try more iterations',
            ha='center', va='center',
            transform=axes[1].transAxes
        )
        axes[1].set_title('Agent Contributions')

    plt.tight_layout()
    plt.savefig(
        'outputs/multi_agent_results.png',
        dpi=150, bbox_inches='tight'
    )
    print("  Chart saved!")
    print("=" * 60)
