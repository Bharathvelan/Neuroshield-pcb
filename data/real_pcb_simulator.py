# real_pcb_simulator.py
# Real electromagnetic physics models
# Scaled to realistic CISPR 32 measurement range

import numpy as np
import pandas as pd

# ─────────────────────────────────────────
# PHYSICAL CONSTANTS
# ─────────────────────────────────────────

C0        = 3e8
MU0       = 4 * np.pi * 1e-7
EPS0      = 8.854e-12
ETA0      = 377.0
MEAS_DIST = 3.0


# ─────────────────────────────────────────
# MODEL 1: Hertzian Dipole
# ─────────────────────────────────────────

def hertzian_dipole_emi(
        trace_length_m,
        frequency_hz,
        current_ma,
        meas_dist_m=MEAS_DIST):

    wavelength   = C0 / frequency_hz
    length_ratio = min(trace_length_m / wavelength, 0.5)
    current_a    = current_ma / 1000.0

    E_field_vm = (
        ETA0 * current_a *
        trace_length_m *
        length_ratio *
        frequency_hz
    ) / (2 * C0 * meas_dist_m)

    if E_field_vm <= 0:
        return -100.0

    E_dbuvm = 20 * np.log10(E_field_vm * 1e6)
    return float(E_dbuvm)


# ─────────────────────────────────────────
# MODEL 2: Transmission Line Mismatch
# ─────────────────────────────────────────

def transmission_line_emi_factor(
        trace_width_mm,
        ground_distance_mm,
        frequency_hz):

    w  = trace_width_mm
    h  = ground_distance_mm
    Er = 4.4

    if w / h <= 1:
        Z0 = (60 / np.sqrt(Er)) * np.log(
            8 * h / w + w / (4 * h)
        )
    else:
        Z0 = (120 * np.pi) / (
            np.sqrt(Er) * (
                w / h + 1.393 +
                0.667 * np.log(w / h + 1.444)
            )
        )

    target_Z  = 50.0
    mismatch  = abs(Z0 - target_Z) / target_Z
    freq_factor = np.log10(
        max(1, frequency_hz / 1e6)
    ) / 3.0

    emi_increase_db = mismatch * freq_factor * 6.0
    return float(emi_increase_db), float(Z0)


# ─────────────────────────────────────────
# MODEL 3: Return Path Loop
# ─────────────────────────────────────────

def return_path_emi(
        trace_length_m,
        ground_distance_m,
        stitching_vias,
        frequency_hz,
        current_ma,
        meas_dist_m=MEAS_DIST):

    via_reduction = max(0.1, 1.0 - stitching_vias * 0.08)
    loop_area_m2  = (
        trace_length_m *
        ground_distance_m *
        via_reduction
    )
    loop_area_cm2 = loop_area_m2 * 1e4
    current_a     = current_ma / 1000.0

    E_field_vm = (
        1.316e-14 *
        (frequency_hz ** 2) *
        loop_area_cm2 *
        current_a
    ) / meas_dist_m

    if E_field_vm <= 0:
        return -100.0

    E_dbuvm = 20 * np.log10(E_field_vm * 1e6)
    return float(E_dbuvm)


# ─────────────────────────────────────────
# MODEL 4: Decap Resonance
# ─────────────────────────────────────────

def decap_effectiveness(
        decap_distance_mm,
        frequency_hz,
        capacitance_nf=100):

    L_parasitic_h = (0.5 * decap_distance_mm) * 1e-9
    C_farad       = capacitance_nf * 1e-9

    if L_parasitic_h > 0 and C_farad > 0:
        SRF_hz = 1.0 / (
            2 * np.pi *
            np.sqrt(L_parasitic_h * C_farad)
        )
    else:
        SRF_hz = 1e9

    if frequency_hz > SRF_hz:
        ratio      = frequency_hz / SRF_hz
        penalty_db = 20 * np.log10(ratio) * 0.5
    else:
        ratio      = SRF_hz / frequency_hz
        penalty_db = -5 * np.log10(ratio)

    return float(penalty_db), float(SRF_hz / 1e6)


# ─────────────────────────────────────────
# MODEL 5: Ground Plane Inductance
# ─────────────────────────────────────────

def ground_plane_effect(
        ground_distance_mm,
        frequency_hz,
        current_ma):

    h_m = ground_distance_mm * 1e-3
    r   = 0.1e-3

    if h_m > 0 and r > 0:
        L_per_m = (MU0 / (2 * np.pi)) * np.log(
            2 * h_m / r
        )
    else:
        L_per_m = 1e-9

    current_a  = current_ma / 1000.0
    dI_dt      = current_a * 2 * np.pi * frequency_hz
    V_noise    = L_per_m * dI_dt

    if V_noise <= 0:
        return 0.0

    noise_db = 20 * np.log10(V_noise * 1000 + 1e-10)
    return float(noise_db * 0.3)


# ─────────────────────────────────────────
# COMBINE ALL MODELS
# Scaled to realistic CISPR 32 range
# ─────────────────────────────────────────

def calculate_real_emi(
        trace_width_mm,
        trace_length_mm,
        ground_distance_mm,
        stitching_vias,
        decap_distance_mm,
        frequency_mhz,
        current_ma=10.0):

    frequency_hz      = frequency_mhz * 1e6
    trace_length_m    = trace_length_mm * 1e-3
    ground_distance_m = ground_distance_mm * 1e-3

    # Model 1
    E_dipole = hertzian_dipole_emi(
        trace_length_m, frequency_hz,
        current_ma, MEAS_DIST
    )

    # Model 2
    tl_increase, Z0 = transmission_line_emi_factor(
        trace_width_mm, ground_distance_mm,
        frequency_hz
    )

    # Model 3
    E_loop = return_path_emi(
        trace_length_m, ground_distance_m,
        stitching_vias, frequency_hz,
        current_ma, MEAS_DIST
    )

    # Model 4
    decap_penalty, SRF = decap_effectiveness(
        decap_distance_mm, frequency_hz
    )

    # Model 5
    gnd_contribution = ground_plane_effect(
        ground_distance_mm, frequency_hz,
        current_ma
    )

    # Scaled combination
    # Tuned to realistic range: 15–65 dBm
    E_total = (
        E_dipole      * 0.05 +
        E_loop        * 0.05 +
        tl_increase   * 0.30 +
        decap_penalty * 0.20 +
        gnd_contribution * 0.10 +
        np.random.normal(0, 1.0)
    )

    # Clip to realistic EMI range
    E_total = float(np.clip(E_total, -20.0, 80.0))
    return E_total


# ─────────────────────────────────────────
# DATASET GENERATOR
# ─────────────────────────────────────────

def generate_real_pcb_samples(
        num_samples=1000,
        seed=42):

    np.random.seed(seed)
    print(f"Generating {num_samples} samples...")

    records = []

    for i in range(num_samples):

        params = {
            'trace_width_mm':
                float(np.random.uniform(0.1, 2.0)),
            'trace_length_mm':
                float(np.random.uniform(5.0, 100.0)),
            'ground_distance_mm':
                float(np.random.uniform(0.1, 2.0)),
            'stitching_vias':
                int(np.random.randint(0, 11)),
            'decap_distance_mm':
                float(np.random.uniform(0.5, 15.0)),
            'frequency_mhz':
                float(np.random.uniform(30, 1000)),
            'current_ma':
                float(np.random.uniform(1.0, 50.0)),
        }

        emi = calculate_real_emi(
            trace_width_mm=params['trace_width_mm'],
            trace_length_mm=params['trace_length_mm'],
            ground_distance_mm=params['ground_distance_mm'],
            stitching_vias=params['stitching_vias'],
            decap_distance_mm=params['decap_distance_mm'],
            frequency_mhz=params['frequency_mhz'],
            current_ma=params['current_ma']
        )

        params['emi_dbm'] = emi
        records.append(params)

        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{num_samples} done")

    df = pd.DataFrame(records)
    return df


# ─────────────────────────────────────────
# TEST
# ─────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 55)
    print("  Real PCB Physics Simulator")
    print("=" * 55)

    LIMIT = 40.0

    print("\n--- Single PCB Tests ---")
    test_cases = [
        {
            'name':               'GOOD PCB',
            'trace_width_mm':     1.5,
            'trace_length_mm':    10.0,
            'ground_distance_mm': 0.2,
            'stitching_vias':     8,
            'decap_distance_mm':  1.0,
            'frequency_mhz':      100.0,
            'current_ma':         5.0
        },
        {
            'name':               'BAD PCB',
            'trace_width_mm':     0.2,
            'trace_length_mm':    90.0,
            'ground_distance_mm': 1.8,
            'stitching_vias':     0,
            'decap_distance_mm':  13.0,
            'frequency_mhz':      900.0,
            'current_ma':         40.0
        },
        {
            'name':               'MEDIUM PCB',
            'trace_width_mm':     0.5,
            'trace_length_mm':    50.0,
            'ground_distance_mm': 0.8,
            'stitching_vias':     3,
            'decap_distance_mm':  5.0,
            'frequency_mhz':      500.0,
            'current_ma':         15.0
        }
    ]

    for tc in test_cases:
        name = tc.pop('name')
        emi  = calculate_real_emi(**tc)
        status = 'PASS' if emi < LIMIT else 'FAIL'
        print(f"  {name}: {emi:.2f} dBm → {status}")

    print("\n--- Generating Dataset ---")
    df = generate_real_pcb_samples(1000)

    print(f"\n  Shape:  {df.shape}")
    print(f"  Min:    {df['emi_dbm'].min():.2f} dBm")
    print(f"  Max:    {df['emi_dbm'].max():.2f} dBm")
    print(f"  Mean:   {df['emi_dbm'].mean():.2f} dBm")

    passing = (df['emi_dbm'] < LIMIT).sum()
    failing = (df['emi_dbm'] >= LIMIT).sum()
    print(f"  Pass:   {passing} ({passing/10:.1f}%)")
    print(f"  Fail:   {failing} ({failing/10:.1f}%)")

    df.to_csv('data/real_pcb_dataset.csv', index=False)
    print(f"\n  Saved to data/real_pcb_dataset.csv")
    print("=" * 55)