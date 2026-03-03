# federated_learning.py
# Phase 4: Federated Learning System
#
# Simulates multiple companies training
# the KAN-PINN model on their own private
# PCB data and sharing only model weights
# (never sharing actual board designs!)

import numpy as np
import torch
import torch.optim as optim
import pandas as pd
import copy
import json
import os
import sys
import matplotlib.pyplot as plt

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
# PART 1: DATA GENERATOR PER COMPANY
# ─────────────────────────────────────────

def generate_company_data(
        company_name,
        num_samples=200,
        seed=42):
    """
    Generates private PCB dataset for
    one company. Each company specializes
    in different types of boards.

    Company A: Consumer electronics
    Company B: Industrial equipment
    Company C: Automotive electronics
    """

    np.random.seed(seed)

    if company_name == 'Company_A':
        freq_range   = (30, 300)
        length_range = (5, 50)
        width_range  = (0.3, 1.5)
        via_range    = (1, 6)
        desc = "Consumer Electronics (30-300 MHz)"

    elif company_name == 'Company_B':
        freq_range   = (200, 600)
        length_range = (20, 80)
        width_range  = (0.2, 1.2)
        via_range    = (0, 8)
        desc = "Industrial Equipment (200-600 MHz)"

    elif company_name == 'Company_C':
        freq_range   = (500, 1000)
        length_range = (30, 100)
        width_range  = (0.1, 1.0)
        via_range    = (2, 10)
        desc = "Automotive (500-1000 MHz)"

    else:
        freq_range   = (30, 1000)
        length_range = (5, 100)
        width_range  = (0.1, 2.0)
        via_range    = (0, 10)
        desc = "Generic"

    print(f"  Generating {company_name} data...")
    print(f"  Specialization: {desc}")

    records = []
    for i in range(num_samples):
        params = {
            'trace_width_mm':
                float(np.random.uniform(
                    *width_range
                )),
            'trace_length_mm':
                float(np.random.uniform(
                    *length_range
                )),
            'ground_distance_mm':
                float(np.random.uniform(
                    0.1, 2.0
                )),
            'stitching_vias':
                int(np.random.randint(
                    via_range[0],
                    via_range[1] + 1
                )),
            'decap_distance_mm':
                float(np.random.uniform(
                    0.5, 15.0
                )),
            'frequency_mhz':
                float(np.random.uniform(
                    *freq_range
                )),
            'current_ma':
                float(np.random.uniform(
                    1.0, 50.0
                )),
        }

        emi = calculate_real_emi(
            trace_width_mm=
                params['trace_width_mm'],
            trace_length_mm=
                params['trace_length_mm'],
            ground_distance_mm=
                params['ground_distance_mm'],
            stitching_vias=
                params['stitching_vias'],
            decap_distance_mm=
                params['decap_distance_mm'],
            frequency_mhz=
                params['frequency_mhz'],
            current_ma=
                params['current_ma']
        )

        params['emi_dbm'] = emi
        records.append(params)

    df = pd.DataFrame(records)
    print(f"  Generated {len(df)} samples")
    print(
        f"  EMI range: "
        f"{df['emi_dbm'].min():.1f} to "
        f"{df['emi_dbm'].max():.1f} dBm"
    )
    return df


# ─────────────────────────────────────────
# PART 2: CONVERT DATA TO TENSORS
# ─────────────────────────────────────────

def prepare_tensors(df):
    """
    Converts company dataframe to
    PyTorch tensors for training.
    """
    feature_vectors = []

    for i in range(len(df)):
        row = df.iloc[i]
        pcb_row = {
            'trace_width_mm':
                float(row['trace_width_mm']),
            'trace_length_mm':
                float(row['trace_length_mm']),
            'ground_distance_mm':
                float(row['ground_distance_mm']),
            'stitching_vias':
                float(row['stitching_vias']),
            'decap_distance_mm':
                float(row['decap_distance_mm']),
            'frequency_mhz':
                float(row['frequency_mhz']),
        }
        try:
            G  = build_pcb_graph(
                pd.Series(pcb_row)
            )
            fv = graph_to_feature_vector(G)
            feature_vectors.append(fv)
        except Exception:
            continue

    X = np.array(feature_vectors)
    y = df['emi_dbm'].values[
        :len(feature_vectors)
    ]

    X_mean = X.mean(axis=0)
    X_std  = X.std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std

    y_mean = y.mean()
    y_std  = y.std() + 1e-8
    y_norm = (y - y_mean) / y_std

    X_t = torch.FloatTensor(X_norm)
    y_t = torch.FloatTensor(
        y_norm
    ).reshape(-1, 1)

    input_size = X.shape[1]
    return X_t, y_t, y_mean, y_std, input_size


# ─────────────────────────────────────────
# PART 3: FEDERATED CLIENT
# ─────────────────────────────────────────

class FederatedClient:
    """
    Represents one company.
    Trains on private data.
    Only shares model weights.
    Never shares actual PCB designs!
    """

    def __init__(self, name, data,
                 input_size, local_epochs=5):
        self.name         = name
        self.input_size   = input_size
        self.local_epochs = local_epochs

        (self.X_train,
         self.y_train,
         self.y_mean,
         self.y_std,
         _) = prepare_tensors(data)

        print(
            f"  Client {name} ready: "
            f"{len(self.X_train)} samples"
        )

    def train_local(self, global_weights):
        """
        Train on private data starting
        from global model weights.
        Returns updated weights only.
        """

        model = KANPINN(
            input_size=self.input_size
        )
        model.load_state_dict(
            copy.deepcopy(global_weights)
        )
        model.train()

        optimizer = optim.Adam(
            model.parameters(),
            lr=0.001,
            weight_decay=1e-5
        )

        total_loss = 0.0

        for epoch in range(self.local_epochs):
            optimizer.zero_grad()
            preds   = model(self.X_train)
            loss, _ = physics_informed_loss(
                preds,
                self.y_train,
                self.X_train
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / self.local_epochs
        return model.state_dict(), avg_loss

    def evaluate(self, global_weights):
        """Evaluate global model on local data"""
        model = KANPINN(
            input_size=self.input_size
        )
        model.load_state_dict(global_weights)
        model.eval()

        with torch.no_grad():
            preds_norm = model(
                self.X_train
            ).numpy().flatten()

        preds_real = (
            preds_norm * self.y_std +
            self.y_mean
        )
        y_real = (
            self.y_train.numpy().flatten() *
            self.y_std + self.y_mean
        )

        mae = float(np.mean(
            np.abs(preds_real - y_real)
        ))
        return mae


# ─────────────────────────────────────────
# PART 4: FEDERATED SERVER
# ─────────────────────────────────────────

class FederatedServer:
    """
    Central server that coordinates
    federated learning using FedAvg.

    FedAvg = weighted average of all
    client model weights.
    """

    def __init__(self, input_size,
                 num_rounds=10):
        self.input_size   = input_size
        self.num_rounds   = num_rounds
        self.clients      = []
        self.global_model = KANPINN(
            input_size=input_size
        )
        self.global_weights = \
            self.global_model.state_dict()
        self.round_history  = []
        self.client_losses  = {}

    def add_client(self, client):
        self.clients.append(client)
        self.client_losses[client.name] = []
        print(f"  Registered: {client.name}")

    def federated_average(
            self,
            client_weights_list,
            client_sizes):
        """
        FedAvg: weighted average of
        all client model weights.
        """

        total_samples = sum(client_sizes)
        avg_weights   = {}
        param_names   = \
            client_weights_list[0].keys()

        for key in param_names:
            weighted_sum = sum(
                client_weights_list[i][key] *
                (
                    client_sizes[i] /
                    total_samples
                )
                for i in range(
                    len(client_weights_list)
                )
            )
            avg_weights[key] = weighted_sum

        return avg_weights

    def train_round(self, round_num):
        """One round of federated learning"""

        print(
            f"\n  Round [{round_num+1:2d}/"
            f"{self.num_rounds}]"
        )

        client_weights = []
        client_sizes   = []
        round_losses   = []

        for client in self.clients:
            weights, loss = \
                client.train_local(
                    self.global_weights
                )
            client_weights.append(weights)
            client_sizes.append(
                len(client.X_train)
            )
            round_losses.append(loss)
            self.client_losses[
                client.name
            ].append(loss)

            print(
                f"    {client.name:<12} | "
                f"Loss: {loss:.4f}"
            )

        # FedAvg
        self.global_weights = \
            self.federated_average(
                client_weights,
                client_sizes
            )
        self.global_model.load_state_dict(
            self.global_weights
        )

        # Evaluate
        avg_maes = []
        for client in self.clients:
            mae = client.evaluate(
                self.global_weights
            )
            avg_maes.append(mae)

        avg_mae  = float(np.mean(avg_maes))
        avg_loss = float(np.mean(round_losses))

        self.round_history.append({
            'round':    round_num + 1,
            'avg_loss': avg_loss,
            'avg_mae':  avg_mae
        })

        print(
            f"    Global MAE: {avg_mae:.4f} dBm"
        )

        return avg_loss, avg_mae

    def train(self):
        """Run all federated rounds"""

        print("\n" + "=" * 60)
        print(
            "  NeuroShield-PCB: "
            "Federated Learning"
        )
        print("=" * 60)
        print(
            f"\n  Clients:  {len(self.clients)}"
        )
        print(
            f"  Rounds:   {self.num_rounds}"
        )
        print(
            "  Strategy: FedAvg"
        )
        print(f"  {'-'*50}")

        for round_num in range(self.num_rounds):
            self.train_round(round_num)

        final_mae = self.round_history[-1][
            'avg_mae'
        ]
        print(f"\n  {'='*50}")
        print("  Federated Training Complete!")
        print(
            f"  Final Global MAE: "
            f"{final_mae:.4f} dBm"
        )
        print(f"  {'='*50}")

        return self.global_model

    def save_global_model(self, path):
        """Save the federated global model"""
        os.makedirs(
            os.path.dirname(path)
            if os.path.dirname(path)
            else 'outputs',
            exist_ok=True
        )
        final_mae = self.round_history[-1][
            'avg_mae'
        ]
        torch.save({
            'model_state_dict':
                self.global_weights,
            'input_size':
                self.input_size,
            'trained_on':
                'federated_real_physics',
            'num_clients':
                len(self.clients),
            'num_rounds':
                self.num_rounds,
            'final_mae':
                final_mae
        }, path)
        print(
            f"\n  Global model saved to {path}"
        )

    def plot_results(self):
        """Plot federated learning progress"""

        fig, axes = plt.subplots(
            1, 3, figsize=(18, 5)
        )

        rounds = [
            r['round']
            for r in self.round_history
        ]
        losses = [
            r['avg_loss']
            for r in self.round_history
        ]
        maes = [
            r['avg_mae']
            for r in self.round_history
        ]

        # Plot 1: Global loss
        axes[0].plot(
            rounds, losses,
            color='blue', linewidth=2,
            marker='o', markersize=6,
            label='Global Loss'
        )
        axes[0].set_xlabel('Round')
        axes[0].set_ylabel('Average Loss')
        axes[0].set_title(
            'Federated Training Loss',
            fontweight='bold'
        )
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Global MAE
        axes[1].plot(
            rounds, maes,
            color='green', linewidth=2,
            marker='s', markersize=6,
            label='Global MAE (dBm)'
        )
        axes[1].set_xlabel('Round')
        axes[1].set_ylabel('MAE (dBm)')
        axes[1].set_title(
            'Global Model Accuracy',
            fontweight='bold'
        )
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # Plot 3: Per-client loss
        colors = [
            '#ff6b6b', '#4ecdc4', '#45b7d1'
        ]
        for i, client in enumerate(
            self.clients
        ):
            c_losses = self.client_losses[
                client.name
            ]
            axes[2].plot(
                range(1, len(c_losses) + 1),
                c_losses,
                color=colors[i],
                linewidth=2,
                marker='o', markersize=5,
                label=client.name
            )
        axes[2].set_xlabel('Round')
        axes[2].set_ylabel('Local Loss')
        axes[2].set_title(
            'Per-Client Training Loss',
            fontweight='bold'
        )
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        plt.suptitle(
            'NeuroShield-PCB: '
            'Federated Learning Results',
            fontsize=14,
            fontweight='bold'
        )
        plt.tight_layout()
        plt.savefig(
            'outputs/federated_results.png',
            dpi=150,
            bbox_inches='tight'
        )
        print(
            "  Plot saved to "
            "outputs/federated_results.png"
        )
        return fig


# ─────────────────────────────────────────
# PART 5: PRIVACY REPORT
# ─────────────────────────────────────────

def generate_privacy_report(clients, server):
    """Shows what was shared vs private"""

    first_mae = server.round_history[0]['avg_mae']
    last_mae  = server.round_history[-1]['avg_mae']

    report = {
        'privacy_summary': {
            'data_shared':    'NEVER',
            'weights_shared': 'YES (anonymized)',
            'board_designs':  'PRIVATE',
            'emi_values':     'PRIVATE',
            'model_weights':  'SHARED'
        },
        'clients': {},
        'global_model': {
            'rounds':      server.num_rounds,
            'final_mae':   float(last_mae),
            'improvement': float(
                first_mae - last_mae
            )
        }
    }

    for client in clients:
        report['clients'][client.name] = {
            'samples':        len(client.X_train),
            'data_shared':    False,
            'weights_shared': True
        }

    return report


# ─────────────────────────────────────────
# MAIN: RUN FEDERATION
# ─────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 60)
    print("  Phase 4: Federated Learning")
    print("=" * 60)

    # Step 1: Generate private datasets
    print(
        "\n[1/5] Generating company datasets..."
    )
    print(
        "  (Each company keeps data private!)\n"
    )

    data_a = generate_company_data(
        'Company_A',
        num_samples=200,
        seed=42
    )
    print()
    data_b = generate_company_data(
        'Company_B',
        num_samples=200,
        seed=123
    )
    print()
    data_c = generate_company_data(
        'Company_C',
        num_samples=200,
        seed=456
    )

    # Step 2: Auto-detect input size
    print(
        "\n[2/5] Creating federated clients..."
    )
    test_row = {
        'trace_width_mm':     0.5,
        'trace_length_mm':    50.0,
        'ground_distance_mm': 0.5,
        'stitching_vias':     3.0,
        'decap_distance_mm':  5.0,
        'frequency_mhz':      500.0,
    }
    G          = build_pcb_graph(
        pd.Series(test_row)
    )
    fv         = graph_to_feature_vector(G)
    input_size = len(fv)
    print(f"  Input size: {input_size}")

    client_a = FederatedClient(
        'Company_A', data_a,
        input_size, local_epochs=5
    )
    client_b = FederatedClient(
        'Company_B', data_b,
        input_size, local_epochs=5
    )
    client_c = FederatedClient(
        'Company_C', data_c,
        input_size, local_epochs=5
    )

    # Step 3: Create server
    print(
        "\n[3/5] Setting up federated server..."
    )
    server = FederatedServer(
        input_size=input_size,
        num_rounds=10
    )
    server.add_client(client_a)
    server.add_client(client_b)
    server.add_client(client_c)

    # Step 4: Run federated training
    print(
        "\n[4/5] Running federated training..."
    )
    global_model = server.train()

    # Step 5: Save results
    print("\n[5/5] Saving results...")

    os.makedirs('outputs', exist_ok=True)

    server.save_global_model(
        'outputs/federated_model.pth'
    )

    privacy_report = generate_privacy_report(
        [client_a, client_b, client_c],
        server
    )

    with open(
        'outputs/federated_report.json', 'w'
    ) as f:
        json.dump(privacy_report, f, indent=2)
    print(
        "  Privacy report saved!"
    )

    server.plot_results()

    # Final summary
    h         = server.round_history
    first_mae = h[0]['avg_mae']
    last_mae  = h[-1]['avg_mae']
    improvement = first_mae - last_mae

    print("\n" + "=" * 60)
    print("  FEDERATED LEARNING SUMMARY")
    print("=" * 60)

    print(f"\n  Companies participated: 3")
    print(
        f"  Company A: "
        f"{len(client_a.X_train)} boards "
        f"(Consumer Electronics)"
    )
    print(
        f"  Company B: "
        f"{len(client_b.X_train)} boards "
        f"(Industrial)"
    )
    print(
        f"  Company C: "
        f"{len(client_c.X_train)} boards "
        f"(Automotive)"
    )

    print(f"\n  Privacy Protection:")
    print(f"  Board designs shared:  NEVER")
    print(f"  EMI values shared:     NEVER")
    print(f"  Model weights shared:  YES")

    print(f"\n  Training Results:")
    print(
        f"  Round 1 MAE:  {first_mae:.4f} dBm"
    )
    print(
        f"  Final MAE:    {last_mae:.4f} dBm"
    )
    print(
        f"  Improvement:  {improvement:.4f} dBm"
    )

    print(f"\n  Output Files:")
    print(f"  outputs/federated_model.pth")
    print(f"  outputs/federated_report.json")
    print(f"  outputs/federated_results.png")
    print("=" * 60)

