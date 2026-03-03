# generative_designer.py
# Phase 7C: Generative Layout Designer
#
# AI generates NEW PCB layouts that are
# predicted to pass EMC compliance!
#
# Method: Variational Autoencoder (VAE)
# - Encoder: PCB params → latent space
# - Decoder: latent space → PCB params
# - Sample from latent space → new designs!
#
# Then filter generated designs by:
# - EMC compliance prediction
# - Physics validity checks
# - Design rule compliance
#
# Used in:
# - Drug discovery (generate new molecules)
# - Chip design (generate circuit layouts)
# - Material science (generate new materials)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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
from real_pcb_simulator import calculate_real_emi


# ─────────────────────────────────────────
# PART 1: VARIATIONAL AUTOENCODER
# ─────────────────────────────────────────

class PCBEncoder(nn.Module):
    """
    Encodes PCB parameters into
    latent space distribution (mu, logvar).
    """

    def __init__(self, input_dim=6,
                 hidden_dim=64,
                 latent_dim=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU()
        )
        self.mu_layer     = nn.Linear(
            hidden_dim // 2, latent_dim
        )
        self.logvar_layer = nn.Linear(
            hidden_dim // 2, latent_dim
        )

    def forward(self, x):
        h      = self.net(x)
        mu     = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        return mu, logvar


class PCBDecoder(nn.Module):
    """
    Decodes latent vector into
    PCB parameters.
    """

    def __init__(self, latent_dim=8,
                 hidden_dim=64,
                 output_dim=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()  # Output in [0,1]
        )

    def forward(self, z):
        return self.net(z)


class PCBVAE(nn.Module):
    """
    Variational Autoencoder for PCB design.

    Key insight:
    - Normal autoencoder: encodes to point
    - VAE: encodes to distribution (mu, sigma)
    - Sample from distribution → new design!

    This lets us:
    1. Learn the space of good PCB designs
    2. Generate new designs by sampling
    3. Interpolate between designs
    4. Find designs near compliance boundary
    """

    def __init__(self, input_dim=6,
                 hidden_dim=64,
                 latent_dim=8):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder    = PCBEncoder(
            input_dim, hidden_dim, latent_dim
        )
        self.decoder    = PCBDecoder(
            latent_dim, hidden_dim, input_dim
        )

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick:
        z = mu + eps * std
        where eps ~ N(0,1)
        Makes backprop through sampling possible!
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z          = self.reparameterize(
            mu, logvar
        )
        recon      = self.decoder(z)
        return recon, mu, logvar

    def generate(self, n_samples=10,
                 device='cpu'):
        """
        Generate new PCB designs by
        sampling from latent space.
        """
        self.eval()
        with torch.no_grad():
            z = torch.randn(
                n_samples, self.latent_dim
            ).to(device)
            generated = self.decoder(z)
        return generated


# ─────────────────────────────────────────
# PART 2: VAE LOSS
# ─────────────────────────────────────────

def vae_loss(recon, original,
             mu, logvar, beta=1.0):
    """
    VAE loss = Reconstruction + KL divergence

    Reconstruction: how well we recreate input
    KL divergence:  how close latent space
                    is to standard normal N(0,1)

    Beta controls balance:
    - Beta=1: standard VAE
    - Beta>1: more disentangled latent space
    """

    # Reconstruction loss (MSE)
    recon_loss = nn.functional.mse_loss(
        recon, original, reduction='mean'
    )

    # KL divergence
    # KL = -0.5 * sum(1 + logvar - mu² - e^logvar)
    kl_loss = -0.5 * torch.mean(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    total = recon_loss + beta * kl_loss
    return total, recon_loss, kl_loss


# ─────────────────────────────────────────
# PART 3: DATA PREPARATION
# ─────────────────────────────────────────

def generate_pcb_dataset(
        n_samples=1000, seed=42):
    """
    Generate diverse PCB parameter dataset.
    Includes both passing and failing designs.
    """
    np.random.seed(seed)

    records = []
    print(
        f"  Generating {n_samples} PCB designs..."
    )

    for i in range(n_samples):
        params = {
            'trace_width_mm':
                float(np.random.uniform(
                    0.1, 2.0
                )),
            'trace_length_mm':
                float(np.random.uniform(
                    5.0, 100.0
                )),
            'ground_distance_mm':
                float(np.random.uniform(
                    0.1, 2.0
                )),
            'stitching_vias':
                float(np.random.randint(
                    0, 11
                )),
            'decap_distance_mm':
                float(np.random.uniform(
                    0.5, 15.0
                )),
            'frequency_mhz':
                float(np.random.uniform(
                    30, 1000
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
                int(params['stitching_vias']),
            decap_distance_mm=
                params['decap_distance_mm'],
            frequency_mhz=
                params['frequency_mhz'],
            current_ma=10.0
        )
        params['emi_dbm'] = emi
        params['passes']  = emi < 40.0
        records.append(params)

        if (i + 1) % 200 == 0:
            print(f"    {i+1}/{n_samples} done")

    df         = pd.DataFrame(records)
    pass_count = df['passes'].sum()
    print(
        f"  Passing designs: {pass_count} "
        f"({pass_count*100//n_samples}%)"
    )
    return df


def prepare_vae_tensors(df):
    """
    Normalize PCB parameters to [0,1]
    for VAE training.
    """
    params = [
        'trace_width_mm',
        'trace_length_mm',
        'ground_distance_mm',
        'stitching_vias',
        'decap_distance_mm',
        'frequency_mhz'
    ]

    # Min-max normalization bounds
    bounds = {
        'trace_width_mm':
            (0.1, 2.0),
        'trace_length_mm':
            (5.0, 100.0),
        'ground_distance_mm':
            (0.1, 2.0),
        'stitching_vias':
            (0.0, 10.0),
        'decap_distance_mm':
            (0.5, 15.0),
        'frequency_mhz':
            (30.0, 1000.0)
    }

    X = np.zeros(
        (len(df), len(params)),
        dtype=np.float32
    )
    for i, p in enumerate(params):
        lo, hi = bounds[p]
        X[:, i] = (
            df[p].values - lo
        ) / (hi - lo + 1e-8)
        X[:, i] = np.clip(X[:, i], 0.0, 1.0)

    return (
        torch.FloatTensor(X),
        bounds,
        params
    )


# ─────────────────────────────────────────
# PART 4: TRAINER
# ─────────────────────────────────────────

def train_vae(X_tensor, latent_dim=8,
              epochs=100, lr=0.001):
    """Train the PCB VAE"""

    print(f"\n  Training PCB VAE...")
    print(f"  Samples:    {len(X_tensor)}")
    print(f"  Latent dim: {latent_dim}")
    print(f"  Epochs:     {epochs}")

    # Split
    split  = int(0.8 * len(X_tensor))
    X_tr   = X_tensor[:split]
    X_val  = X_tensor[split:]

    # Model
    model = PCBVAE(
        input_dim=6,
        hidden_dim=64,
        latent_dim=latent_dim
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=1e-5
    )
    scheduler = \
        optim.lr_scheduler.CosineAnnealingLR(
            optimizer, epochs
        )

    train_losses = []
    val_losses   = []
    best_val     = float('inf')
    best_state   = None

    for epoch in range(epochs):
        # Train
        model.train()
        optimizer.zero_grad()
        recon, mu, logvar = model(X_tr)
        loss, recon_l, kl_l = vae_loss(
            recon, X_tr, mu, logvar, beta=0.5
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0
        )
        optimizer.step()
        scheduler.step()

        # Validate
        model.eval()
        with torch.no_grad():
            recon_v, mu_v, lv_v = model(X_val)
            loss_v, _, _ = vae_loss(
                recon_v, X_val,
                mu_v, lv_v, beta=0.5
            )

        train_losses.append(float(loss.item()))
        val_losses.append(float(loss_v.item()))

        if loss_v.item() < best_val:
            best_val   = loss_v.item()
            best_state = {
                k: v.clone()
                for k, v
                in model.state_dict().items()
            }

        if (epoch + 1) % 20 == 0:
            print(
                f"    Epoch [{epoch+1:3d}/"
                f"{epochs}] | "
                f"Loss: {loss.item():.4f} | "
                f"Recon: {recon_l.item():.4f} | "
                f"KL: {kl_l.item():.4f}"
            )

    model.load_state_dict(best_state)
    print(
        f"\n  Best Val Loss: {best_val:.4f}"
    )
    return model, train_losses, val_losses


# ─────────────────────────────────────────
# PART 5: GENERATIVE DESIGNER
# ─────────────────────────────────────────

class GenerativeDesigner:
    """
    Uses trained VAE to generate
    new PCB designs that pass EMC!

    Workflow:
    1. Sample from latent space
    2. Decode to PCB parameters
    3. Denormalize parameters
    4. Evaluate with KAN-PINN
    5. Keep only passing designs
    6. Rank by EMC margin
    """

    def __init__(self, vae, kan_model,
                 input_size, bounds, params):
        self.vae        = vae
        self.kan_model  = kan_model
        self.input_size = input_size
        self.bounds     = bounds
        self.params     = params
        self.LIMIT      = 40.0

    def denormalize(self, X_norm):
        """Convert [0,1] back to real values"""
        result = {}
        for i, p in enumerate(self.params):
            lo, hi = self.bounds[p]
            val = float(
                X_norm[i] * (hi - lo) + lo
            )
            # Clip to valid range
            val = max(lo, min(hi, val))
            result[p] = val

        # Round vias to integer
        result['stitching_vias'] = float(
            round(result['stitching_vias'])
        )
        return result

    def evaluate_design(self, pcb_params):
        """Evaluate EMI for a design"""
        pcb_series = pd.Series(pcb_params)
        try:
            G  = build_pcb_graph(pcb_series)
            fv = graph_to_feature_vector(G)
            fv = fv[:self.input_size]
            X  = torch.FloatTensor(
                fv.reshape(1, -1)
            )
            with torch.no_grad():
                emi = self.kan_model(X).item()
            return float(emi)
        except Exception:
            return 999.0

    def generate_compliant_designs(
            self,
            n_generate=1000,
            target_margin=5.0):
        """
        Generate designs predicted to
        pass EMC with good margin.

        n_generate: how many to sample
        target_margin: margin below limit (dBm)
        """

        print(
            f"\n  Generating {n_generate} "
            f"candidate designs..."
        )

        # Generate from VAE
        self.vae.eval()
        with torch.no_grad():
            z = torch.randn(
                n_generate, self.vae.latent_dim
            )
            generated_norm = \
                self.vae.decoder(z).numpy()

        # Denormalize and evaluate
        designs = []
        passing = []

        for i in range(n_generate):
            pcb = self.denormalize(
                generated_norm[i]
            )
            emi = self.evaluate_design(pcb)
            pcb['emi_dbm']  = round(emi, 2)
            pcb['margin']   = round(
                self.LIMIT - emi, 2
            )
            pcb['passes']   = emi < self.LIMIT
            designs.append(pcb)

            if emi < (self.LIMIT - target_margin):
                passing.append(pcb)

            if (i + 1) % 200 == 0:
                print(
                    f"    Evaluated {i+1}/"
                    f"{n_generate} | "
                    f"Passing so far: "
                    f"{len(passing)}"
                )

        # Sort by margin (best first)
        passing.sort(
            key=lambda x: x['emi_dbm']
        )

        pass_rate = len(passing) / n_generate

        print(
            f"\n  Total generated: {n_generate}"
        )
        print(
            f"  Passing designs: {len(passing)} "
            f"({pass_rate*100:.1f}%)"
        )

        if passing:
            best = passing[0]
            print(
                f"  Best design EMI: "
                f"{best['emi_dbm']:.2f} dBm"
            )
            print(
                f"  Best margin:     "
                f"{best['margin']:.2f} dBm"
            )

        return passing, designs

    def interpolate_designs(
            self, pcb_a, pcb_b,
            n_steps=10):
        """
        Interpolate between two PCB designs
        in latent space.

        This finds a smooth path between
        two designs — useful for gradually
        modifying a design while staying
        compliant!
        """

        # Normalize both designs
        def normalize(pcb):
            vec = np.zeros(6, dtype=np.float32)
            for i, p in enumerate(self.params):
                lo, hi = self.bounds[p]
                vec[i] = float(
                    (pcb[p] - lo) /
                    (hi - lo + 1e-8)
                )
            return vec

        vec_a = normalize(pcb_a)
        vec_b = normalize(pcb_b)

        X_a = torch.FloatTensor(
            vec_a.reshape(1, -1)
        )
        X_b = torch.FloatTensor(
            vec_b.reshape(1, -1)
        )

        # Encode both to latent space
        self.vae.eval()
        with torch.no_grad():
            mu_a, _ = self.vae.encoder(X_a)
            mu_b, _ = self.vae.encoder(X_b)

        # Interpolate in latent space
        interpolated = []
        alphas       = np.linspace(0, 1, n_steps)

        for alpha in alphas:
            z = (
                (1 - alpha) * mu_a +
                alpha       * mu_b
            )
            with torch.no_grad():
                decoded = self.vae.decoder(
                    z
                ).numpy()[0]
            pcb = self.denormalize(decoded)
            emi = self.evaluate_design(pcb)
            pcb['emi_dbm'] = round(emi, 2)
            pcb['alpha']   = round(
                float(alpha), 2
            )
            pcb['passes']  = emi < self.LIMIT
            interpolated.append(pcb)

        return interpolated


# ─────────────────────────────────────────
# PART 6: VISUALIZER
# ─────────────────────────────────────────

def plot_generative_results(
        passing_designs,
        all_designs,
        train_losses,
        val_losses,
        interp=None):
    """Plot generative design results"""

    fig, axes = plt.subplots(
        2, 2, figsize=(14, 10)
    )

    # Plot 1: Training loss
    axes[0, 0].plot(
        train_losses, color='blue',
        linewidth=1.5, label='Train Loss'
    )
    axes[0, 0].plot(
        val_losses, color='red',
        linewidth=1.5, linestyle='--',
        label='Val Loss'
    )
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('VAE Loss')
    axes[0, 0].set_title(
        'VAE Training Loss',
        fontweight='bold'
    )
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: EMI distribution of designs
    all_emis     = [
        d['emi_dbm'] for d in all_designs
    ]
    passing_emis = [
        d['emi_dbm'] for d in passing_designs
    ]

    axes[0, 1].hist(
        all_emis, bins=40,
        color='lightblue', alpha=0.7,
        label=f'All ({len(all_designs)})',
        edgecolor='blue'
    )
    if passing_emis:
        axes[0, 1].hist(
            passing_emis, bins=20,
            color='green', alpha=0.7,
            label=(
                f'Passing '
                f'({len(passing_designs)})'
            ),
            edgecolor='darkgreen'
        )
    axes[0, 1].axvline(
        x=40.0, color='red',
        linestyle='--', linewidth=2,
        label='CISPR 32 Limit'
    )
    axes[0, 1].set_xlabel('Predicted EMI (dBm)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title(
        'Generated Design Distribution',
        fontweight='bold'
    )
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Top designs comparison
    if passing_designs:
        top = passing_designs[:min(
            10, len(passing_designs)
        )]
        names  = [f"D{i+1}" for i in range(
            len(top)
        )]
        emis   = [d['emi_dbm'] for d in top]
        colors = [
            'green' if e < 35 else
            'lightgreen'
            for e in emis
        ]
        axes[1, 0].bar(
            names, emis,
            color=colors,
            edgecolor='black', alpha=0.8
        )
        axes[1, 0].axhline(
            y=40.0, color='red',
            linestyle='--', linewidth=2,
            label='Limit (40 dBm)'
        )
        axes[1, 0].set_xlabel('Generated Design')
        axes[1, 0].set_ylabel('EMI (dBm)')
        axes[1, 0].set_title(
            'Top 10 Generated Designs',
            fontweight='bold'
        )
        axes[1, 0].legend()
        axes[1, 0].grid(
            True, alpha=0.3, axis='y'
        )

    # Plot 4: Interpolation
    if interp:
        alphas = [d['alpha'] for d in interp]
        emis_i = [d['emi_dbm'] for d in interp]
        colors_i = [
            'green' if d['passes'] else 'red'
            for d in interp
        ]
        axes[1, 1].plot(
            alphas, emis_i,
            color='blue', linewidth=2,
            label='EMI along path'
        )
        axes[1, 1].scatter(
            alphas, emis_i,
            c=colors_i, s=80, zorder=5
        )
        axes[1, 1].axhline(
            y=40.0, color='red',
            linestyle='--', linewidth=2,
            label='CISPR 32 Limit'
        )
        axes[1, 1].set_xlabel(
            'Interpolation (0=Design A, '
            '1=Design B)'
        )
        axes[1, 1].set_ylabel('EMI (dBm)')
        axes[1, 1].set_title(
            'Latent Space Interpolation',
            fontweight='bold'
        )
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle(
        'NeuroShield-PCB: '
        'Generative Layout Designer',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    plt.savefig(
        'outputs/generative_results.png',
        dpi=150, bbox_inches='tight'
    )
    print(
        "  Chart saved to "
        "outputs/generative_results.png"
    )
    return fig


# ─────────────────────────────────────────
# MAIN: TRAIN AND GENERATE
# ─────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 60)
    print(
        "  Phase 7C: Generative Layout Designer"
    )
    print("=" * 60)

    # Load KAN-PINN
    print("\nLoading KAN-PINN model...")
    checkpoint = torch.load(
        'outputs/kan_pinn_model.pth',
        weights_only=False
    )
    input_size = checkpoint['input_size']
    kan_model  = KANPINN(input_size=input_size)
    kan_model.load_state_dict(
        checkpoint['model_state_dict']
    )
    kan_model.eval()
    print(f"KAN-PINN loaded!")

    # Generate dataset
    print("\n[1/5] Generating PCB dataset...")
    df = generate_pcb_dataset(
        n_samples=1000, seed=42
    )

    # Prepare tensors
    print("\n[2/5] Preparing tensors...")
    X_tensor, bounds, params = \
        prepare_vae_tensors(df)
    print(f"  Tensor shape: {X_tensor.shape}")

    # Train VAE
    print("\n[3/5] Training VAE...")
    vae, train_losses, val_losses = train_vae(
        X_tensor,
        latent_dim=8,
        epochs=100,
        lr=0.001
    )

    # Create designer
    designer = GenerativeDesigner(
        vae, kan_model, input_size,
        bounds, params
    )

    # Generate compliant designs
    print("\n[4/5] Generating compliant designs...")
    passing, all_designs = \
        designer.generate_compliant_designs(
            n_generate=1000,
            target_margin=3.0
        )

    # Interpolation demo
    print("\n[5/5] Latent space interpolation...")
    if len(passing) >= 2:
        interp = designer.interpolate_designs(
            passing[0],
            passing[-1],
            n_steps=10
        )
        print(
            f"  Interpolated {len(interp)} "
            f"designs between best and worst"
        )
    else:
        interp = None

    # Save
    os.makedirs('outputs', exist_ok=True)
    torch.save({
        'vae_state_dict': vae.state_dict(),
        'bounds':         bounds,
        'params':         params,
        'latent_dim':     8
    }, 'outputs/vae_model.pth')
    print(
        "\n  VAE saved to outputs/vae_model.pth"
    )

    # Save top designs
    top_10 = passing[:min(10, len(passing))]
    with open(
        'outputs/generated_designs.json', 'w'
    ) as f:
        json.dump({
            'total_generated': len(all_designs),
            'total_passing':   len(passing),
            'pass_rate':
                len(passing) / len(all_designs),
            'top_designs':     top_10
        }, f, indent=2)

    # Plot
    fig = plot_generative_results(
        passing, all_designs,
        train_losses, val_losses,
        interp
    )
    plt.close()

    # Summary
    print("\n" + "=" * 60)
    print("  GENERATIVE DESIGNER SUMMARY")
    print("=" * 60)
    print(
        f"  Designs generated: {len(all_designs)}"
    )
    print(
        f"  Passing designs:   {len(passing)}"
    )
    print(
        f"  Pass rate:         "
        f"{len(passing)*100//len(all_designs)}%"
    )
    if passing:
        print(
            f"  Best EMI:          "
            f"{passing[0]['emi_dbm']:.2f} dBm"
        )
        print(
            f"  Best margin:       "
            f"{passing[0]['margin']:.2f} dBm"
        )
        print(
            f"\n  Best Design Parameters:"
        )
        best = passing[0]
        for p in params:
            print(
                f"    {p:<25}: "
                f"{best[p]:.3f}"
            )
    print(
        f"\n  Output Files:"
    )
    print(f"  outputs/vae_model.pth")
    print(f"  outputs/generated_designs.json")
    print(f"  outputs/generative_results.png")
    print("=" * 60)
