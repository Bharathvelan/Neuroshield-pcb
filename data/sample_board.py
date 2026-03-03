# sample_board.py
# This file generates fake PCB designs with EMI values for training

import numpy as np
import pandas as pd

def generate_pcb_samples(num_samples=1000):
    """
    Creates fake PCB designs.
    Each design has features like trace width, trace length, etc.
    And a label: the EMI value (how much radiation it produces)
    """

    np.random.seed(42)  # makes results repeatable

    data = {
        # Trace width in millimeters (thicker = less EMI usually)
        'trace_width_mm':       np.random.uniform(0.1, 2.0, num_samples),

        # Trace length in millimeters (longer = more EMI)
        'trace_length_mm':      np.random.uniform(5.0, 100.0, num_samples),

        # Distance to ground plane in mm (closer = less EMI)
        'ground_distance_mm':   np.random.uniform(0.1, 2.0, num_samples),

        # Number of stitching vias nearby (more = less EMI)
        'stitching_vias':       np.random.randint(0, 10, num_samples),

        # Decoupling capacitor distance in mm (closer = less EMI)
        'decap_distance_mm':    np.random.uniform(0.5, 15.0, num_samples),

        # Signal frequency in MHz (higher = more EMI)
        'frequency_mhz':        np.random.uniform(50, 3000, num_samples),
    }

    df = pd.DataFrame(data)

    # Calculate EMI value based on physics rules (simplified)
    # Higher frequency + longer trace + less ground = more EMI
    df['emi_dbm'] = (
        0.05  * df['frequency_mhz'] +
        0.30  * df['trace_length_mm'] -
        5.00  * df['trace_width_mm'] -
        3.00  * df['stitching_vias'] +
        2.00  * df['decap_distance_mm'] -
        10.00 * df['ground_distance_mm'] +
        np.random.normal(0, 2, num_samples)  # small random noise
    )

    return df

# Test it immediately
if __name__ == "__main__":
    df = generate_pcb_samples(1000)
    print("Generated", len(df), "PCB samples")
    print("\nFirst 5 samples:")
    print(df.head())
    print("\nEMI Statistics:")
    print(df['emi_dbm'].describe())
