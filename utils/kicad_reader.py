# kicad_reader.py
# Reads real KiCad PCB files using pure Python
# No pcbnew needed — works everywhere!

import re
import os
import sys
import json
import numpy as np

sys.path.append('data')
sys.path.append('models')
sys.path.append('utils')


# ─────────────────────────────────────────
# PART 1: KiCad File Parser
# ─────────────────────────────────────────

class KiCadPCBReader:
    """
    Reads .kicad_pcb files using pure Python.
    Extracts all traces, vias, and components.
    No KiCad installation needed for parsing!
    """

    def __init__(self, filepath):
        self.filepath   = filepath
        self.traces     = []
        self.vias       = []
        self.components = []
        self.nets       = {}
        self.board_info = {}

    def read(self):
        """Read and parse the KiCad PCB file"""
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(
                f"PCB file not found: {self.filepath}"
            )

        with open(self.filepath, 'r',
                  encoding='utf-8') as f:
            content = f.read()

        print(f"  Reading: {self.filepath}")
        print(f"  File size: {len(content)} chars")

        self._parse_nets(content)
        self._parse_traces(content)
        self._parse_vias(content)
        self._parse_components(content)

        return self

    def _parse_nets(self, content):
        """Extract net definitions"""
        pattern = r'\(net\s+(\d+)\s+"([^"]*)"\)'
        matches = re.findall(pattern, content)
        for net_id, net_name in matches:
            self.nets[int(net_id)] = net_name
        print(f"  Found {len(self.nets)} nets")

    def _parse_traces(self, content):
        """Extract all trace segments"""
        pattern = (
            r'\(segment\s+'
            r'\(start\s+([\d.+-]+)\s+([\d.+-]+)\)\s+'
            r'\(end\s+([\d.+-]+)\s+([\d.+-]+)\)\s+'
            r'\(width\s+([\d.]+)\)\s+'
            r'\(layer\s+"([^"]+)"\)\s+'
            r'\(net\s+(\d+)\)'
        )
        matches = re.findall(pattern, content)

        for m in matches:
            x1, y1, x2, y2 = (
                float(m[0]), float(m[1]),
                float(m[2]), float(m[3])
            )
            width  = float(m[4])
            layer  = m[5]
            net_id = int(m[6])

            # Calculate trace length
            length = np.sqrt(
                (x2 - x1)**2 + (y2 - y1)**2
            )

            self.traces.append({
                'x1':      x1,
                'y1':      y1,
                'x2':      x2,
                'y2':      y2,
                'width':   width,
                'length':  length,
                'layer':   layer,
                'net_id':  net_id,
                'net_name': self.nets.get(
                    net_id, f'net_{net_id}'
                )
            })

        print(f"  Found {len(self.traces)} traces")

    def _parse_vias(self, content):
        """Extract all vias"""
        pattern = (
            r'\(via\s+'
            r'\(at\s+([\d.+-]+)\s+([\d.+-]+)\)\s+'
            r'\(size\s+([\d.]+)\)\s+'
            r'\(drill\s+([\d.]+)\)'
        )
        matches = re.findall(pattern, content)

        for m in matches:
            self.vias.append({
                'x':     float(m[0]),
                'y':     float(m[1]),
                'size':  float(m[2]),
                'drill': float(m[3])
            })

        print(f"  Found {len(self.vias)} vias")

    def _parse_components(self, content):
        """Extract component placements"""
        pattern = (
            r'\(footprint\s+"([^"]+)"\s+'
            r'\(layer\s+"([^"]+)"\)\s+'
            r'\(at\s+([\d.+-]+)\s+([\d.+-]+)\)'
        )
        matches = re.findall(pattern, content)

        for m in matches:
            self.components.append({
                'footprint': m[0],
                'layer':     m[1],
                'x':         float(m[2]),
                'y':         float(m[3])
            })

        print(f"  Found {len(self.components)} "
              f"components")

    def get_summary(self):
        """Returns summary of board"""
        return {
            'total_traces':     len(self.traces),
            'total_vias':       len(self.vias),
            'total_components': len(self.components),
            'total_nets':       len(self.nets),
            'nets':             self.nets
        }


# ─────────────────────────────────────────
# PART 2: EMI Risk Analyzer
# Converts KiCad data to EMI parameters
# ─────────────────────────────────────────

class PCBEMIAnalyzer:
    """
    Takes parsed KiCad board data and
    analyzes each trace for EMI risk.
    """

    def __init__(self, reader, model,
                 input_size, frequency_mhz=500):
        self.reader        = reader
        self.model         = model
        self.input_size    = input_size
        self.frequency_mhz = frequency_mhz
        self.results       = []

    def analyze_all_traces(self):
        """Analyze EMI risk for every trace"""
        from graph_builder import (
            build_pcb_graph,
            graph_to_feature_vector
        )
        import pandas as pd
        import torch

        print(f"\n  Analyzing {len(self.reader.traces)}"
              f" traces for EMI...")

        LIMIT = 40.0

        for i, trace in enumerate(
            self.reader.traces
        ):
            # Count nearby vias
            nearby_vias = self._count_nearby_vias(
                trace, radius=5.0
            )

            # Build PCB parameters from trace data
            pcb_params = {
                'trace_width_mm':
                    float(trace['width']),
                'trace_length_mm':
                    float(trace['length']),
                'ground_distance_mm':
                    0.2 if 'Cu' in trace['layer']
                    else 1.0,
                'stitching_vias':
                    float(nearby_vias),
                'decap_distance_mm':
                    self._estimate_decap_distance(
                        trace
                    ),
                'frequency_mhz':
                    float(self.frequency_mhz)
            }

            # Predict EMI using our AI model
            try:
                pcb_series = pd.Series(pcb_params)
                G  = build_pcb_graph(pcb_series)
                fv = graph_to_feature_vector(G)
                fv = fv[:self.input_size]
                X  = torch.FloatTensor(
                    fv.reshape(1, -1)
                )
                with torch.no_grad():
                    emi = self.model(X).item()
            except Exception:
                emi = 50.0  # default if error

            # Risk level
            if emi < 30:
                risk = 'LOW'
                color = 'GREEN'
            elif emi < LIMIT:
                risk = 'MEDIUM'
                color = 'YELLOW'
            else:
                risk = 'HIGH'
                color = 'RED'

            self.results.append({
                'trace_id':  i + 1,
                'net_name':  trace['net_name'],
                'layer':     trace['layer'],
                'width_mm':  trace['width'],
                'length_mm': round(
                    trace['length'], 2
                ),
                'emi_dbm':   round(emi, 2),
                'risk':      risk,
                'color':     color,
                'status':
                    'PASS' if emi < LIMIT else 'FAIL'
            })

        return self.results

    def _count_nearby_vias(self, trace,
                            radius=5.0):
        """Count vias near this trace"""
        mid_x = (trace['x1'] + trace['x2']) / 2
        mid_y = (trace['y1'] + trace['y2']) / 2
        count = 0
        for via in self.reader.vias:
            dist = np.sqrt(
                (via['x'] - mid_x)**2 +
                (via['y'] - mid_y)**2
            )
            if dist < radius:
                count += 1
        return count

    def _estimate_decap_distance(self, trace):
        """Estimate distance to nearest component"""
        if not self.reader.components:
            return 5.0
        mid_x = (trace['x1'] + trace['x2']) / 2
        mid_y = (trace['y1'] + trace['y2']) / 2
        min_dist = float('inf')
        for comp in self.reader.components:
            dist = np.sqrt(
                (comp['x'] - mid_x)**2 +
                (comp['y'] - mid_y)**2
            )
            if dist < min_dist:
                min_dist = dist
        return float(
            np.clip(min_dist * 0.1, 0.5, 15.0)
        )

    def get_board_summary(self):
        """Overall board EMI summary"""
        if not self.results:
            return {}

        emis     = [r['emi_dbm'] for r in self.results]
        high     = sum(
            1 for r in self.results
            if r['risk'] == 'HIGH'
        )
        medium   = sum(
            1 for r in self.results
            if r['risk'] == 'MEDIUM'
        )
        low      = sum(
            1 for r in self.results
            if r['risk'] == 'LOW'
        )
        passing  = sum(
            1 for r in self.results
            if r['status'] == 'PASS'
        )

        return {
            'total_traces':   len(self.results),
            'max_emi':        round(max(emis), 2),
            'min_emi':        round(min(emis), 2),
            'avg_emi':        round(
                np.mean(emis), 2
            ),
            'high_risk':      high,
            'medium_risk':    medium,
            'low_risk':       low,
            'passing_traces': passing,
            'failing_traces':
                len(self.results) - passing,
            'board_status':
                'PASS' if max(emis) < 40 else 'FAIL'
        }


# ─────────────────────────────────────────
# PART 3: Test Everything
# ─────────────────────────────────────────

if __name__ == "__main__":

    print("=" * 55)
    print("  NeuroShield-PCB: KiCad File Reader")
    print("=" * 55)

    # Step 1: Read PCB file
    print("\n--- Reading PCB File ---")
    reader = KiCadPCBReader(
        'data/sample_board.kicad_pcb'
    )
    reader.read()

    # Step 2: Show summary
    summary = reader.get_summary()
    print(f"\n  Board Summary:")
    print(f"  Traces:     {summary['total_traces']}")
    print(f"  Vias:       {summary['total_vias']}")
    print(f"  Components: {summary['total_components']}")
    print(f"  Nets:       {summary['total_nets']}")

    if reader.nets:
        print(f"\n  Net Names:")
        for net_id, name in reader.nets.items():
            if name:
                print(f"    [{net_id}] {name}")

    # Step 3: Load AI model
    print("\n--- Loading AI Model ---")
    import torch
    checkpoint = torch.load(
        'outputs/kan_pinn_model.pth',
        weights_only=False
    )
    input_size = checkpoint['input_size']

    from kan_pinn import KANPINN
    model = KANPINN(input_size=input_size)
    model.load_state_dict(
        checkpoint['model_state_dict']
    )
    model.eval()
    print(f"  Model loaded! Input size: {input_size}")

    # Step 4: Analyze EMI
    print("\n--- Analyzing EMI per Trace ---")
    analyzer = PCBEMIAnalyzer(
        reader, model, input_size,
        frequency_mhz=500
    )
    results = analyzer.analyze_all_traces()

    # Step 5: Show results
    print(f"\n  {'ID':>4} {'Net':<8} {'Layer':<8}"
          f" {'W(mm)':>6} {'L(mm)':>7}"
          f" {'EMI':>8} {'Risk':<8} Status")
    print(f"  {'-'*60}")

    for r in results:
        print(
            f"  {r['trace_id']:>4} "
            f"{r['net_name']:<8} "
            f"{r['layer']:<8} "
            f"{r['width_mm']:>6.2f} "
            f"{r['length_mm']:>7.2f} "
            f"{r['emi_dbm']:>7.1f}dBm "
            f"{r['risk']:<8} "
            f"{r['status']}"
        )

    # Step 6: Board summary
    board = analyzer.get_board_summary()
    print(f"\n  {'='*50}")
    print(f"  BOARD EMI SUMMARY")
    print(f"  {'='*50}")
    print(f"  Max EMI:       {board['max_emi']} dBm")
    print(f"  Avg EMI:       {board['avg_emi']} dBm")
    print(f"  High Risk:     {board['high_risk']} traces")
    print(f"  Medium Risk:   {board['medium_risk']} traces")
    print(f"  Low Risk:      {board['low_risk']} traces")
    print(f"  Board Status:  {board['board_status']}")
    print(f"  {'='*50}")

    # Save results
    os.makedirs('outputs', exist_ok=True)
    with open(
        'outputs/kicad_analysis.json', 'w'
    ) as f:
        json.dump({
            'board_summary': board,
            'trace_results': results
        }, f, indent=2)
    print(f"\n  Results saved to "
          f"outputs/kicad_analysis.json")
    print("=" * 55)
