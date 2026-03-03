# app.py - COMPLETE FINAL DASHBOARD
# 9 Tabs: EMI + Root Cause + Optimizer +
#         Multi-Agent + Report + Batch +
#         KiCad + Federated + Digital Twin
# Run with: streamlit run app.py

import streamlit as st
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import sys

sys.path.append('data')
sys.path.append('models')
sys.path.append('utils')

from graph_builder import (
    build_pcb_graph,
    graph_to_feature_vector
)
from kan_pinn import KANPINN
from drl_optimizer import calculate_emi, optimize_pcb

# ─────────────────────────────────────────
# PAGE SETUP
# ─────────────────────────────────────────

st.set_page_config(
    page_title="NeuroShield-PCB",
    page_icon="🛡️",
    layout="wide"
)

st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────

@st.cache_resource
def load_model():
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
    return model, input_size


@st.cache_resource
def load_norm_params():
    try:
        y_mean = float(
            np.load('outputs/y_mean.npy')[0]
        )
        y_std = float(
            np.load('outputs/y_std.npy')[0]
        )
        return y_mean, y_std
    except Exception:
        return None, None

# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────

st.markdown(
    '<div class="main-title">'
    '🛡️ NeuroShield-PCB</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">'
    'KAN-PINN + Physics-Guided Optimizer + '
    'Multi-Agent DRL + Federated Learning + '
    'Digital Twin + KiCad Integration'
    '</div>',
    unsafe_allow_html=True
)
st.markdown("---")

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────

st.sidebar.title("⚙️ PCB Parameters")
st.sidebar.markdown("Adjust your PCB design:")

trace_width = st.sidebar.slider(
    "Trace Width (mm)", 0.1, 2.0, 0.5, 0.1
)
trace_length = st.sidebar.slider(
    "Trace Length (mm)", 5.0, 100.0, 50.0, 1.0
)
ground_distance = st.sidebar.slider(
    "Ground Distance (mm)", 0.1, 2.0, 0.5, 0.1
)
stitching_vias = st.sidebar.slider(
    "Stitching Vias", 0, 10, 3, 1
)
decap_distance = st.sidebar.slider(
    "Decap Distance (mm)", 0.5, 15.0, 5.0, 0.5
)
frequency = st.sidebar.slider(
    "Frequency (MHz)", 30, 1000, 500, 10
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🎯 Compliance Standard")
standard = st.sidebar.selectbox(
    "Select Standard",
    [
        "CISPR 32 Class B (40 dBm)",
        "FCC Part 15B (40 dBm)",
        "MIL-STD-461 (30 dBm)"
    ]
)
LIMIT = 30.0 if "MIL" in standard else 40.0

st.sidebar.markdown("---")
st.sidebar.markdown("### 📂 Upload CSV")
uploaded_file = st.sidebar.file_uploader(
    "Upload PCB designs (CSV)",
    type=['csv']
)

# ─────────────────────────────────────────
# LOAD MODEL AND PARAMS
# ─────────────────────────────────────────

model, input_size = load_model()
y_mean, y_std     = load_norm_params()

pcb_row = {
    'trace_width_mm':     trace_width,
    'trace_length_mm':    trace_length,
    'ground_distance_mm': ground_distance,
    'stitching_vias':     float(stitching_vias),
    'decap_distance_mm':  decap_distance,
    'frequency_mhz':      float(frequency)
}

predicted_emi = calculate_emi(
    pcb_row, model, input_size
)

# ─────────────────────────────────────────
# MODEL INFO BANNER
# ─────────────────────────────────────────

checkpoint = torch.load(
    'outputs/kan_pinn_model.pth',
    weights_only=False
)
trained_on = checkpoint.get(
    'trained_on', 'formula_data'
)
mae_dbm  = checkpoint.get('mae_dbm', None)
comp_acc = checkpoint.get('compliance_acc', None)

if trained_on == 'real_physics_data':
    st.success(
        "✅ Model trained on REAL "
        "electromagnetic physics data!"
    )
    ci1, ci2, ci3 = st.columns(3)
    with ci1:
        st.info(
            f"🎯 Accuracy: ±{mae_dbm:.2f} dBm"
            if mae_dbm
            else "🎯 Real Physics Model"
        )
    with ci2:
        st.info(
            f"✅ Compliance: {comp_acc:.1f}%"
            if comp_acc
            else "✅ Physics Trained"
        )
    with ci3:
        st.info("📡 5 Physics Models Active")
    st.markdown("---")
else:
    st.warning(
        "⚠️ Model on formula data. "
        "Run train_real.py for better accuracy."
    )

# ─────────────────────────────────────────
# 9 TABS
# ─────────────────────────────────────────

(tab1, tab2, tab3, tab4,
 tab5, tab6, tab7, tab8,
 tab9, tab10) = st.tabs([
    "📊 EMI Analysis",
    "🔍 Root Cause",
    "⚡ Optimizer",
    "🤖 Multi-Agent",
    "📋 Report",
    "📂 Batch Analysis",
    "🔌 KiCad Analysis",
    "🌐 Federated Learning",
    "🔬 Digital Twin",
    "🧠 Advanced AI"
])

# ════════════════════════════════════════
# TAB 1: EMI ANALYSIS
# ════════════════════════════════════════

with tab1:
    st.subheader("📊 Real-Time EMI Analysis")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Predicted EMI",
            f"{predicted_emi:.2f} dBm"
        )
    with col2:
        st.metric(
            "Compliance Limit",
            f"{LIMIT:.1f} dBm"
        )
    with col3:
        margin = abs(predicted_emi - LIMIT)
        over   = predicted_emi > LIMIT
        st.metric(
            "Margin",
            f"{margin:.2f} dBm",
            delta=(
                f"{'Over' if over else 'Under'}"
                f" limit"
            )
        )
    with col4:
        if predicted_emi <= LIMIT:
            st.success("✅ PASS")
        else:
            st.error("❌ FAIL")

    st.markdown("---")

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.barh(
        ['EMI Level'], [LIMIT],
        color='lightgreen', height=0.4,
        label='Safe Zone'
    )
    ax.barh(
        ['EMI Level'],
        [max(0, predicted_emi - LIMIT)],
        left=LIMIT,
        color='salmon', height=0.4,
        label='Danger Zone'
    )
    ax.axvline(
        x=predicted_emi,
        color='darkred', linewidth=3,
        linestyle='--',
        label=f'Your PCB: {predicted_emi:.1f} dBm'
    )
    ax.axvline(
        x=LIMIT,
        color='darkgreen', linewidth=2,
        label=f'Limit: {LIMIT} dBm'
    )
    ax.set_xlabel('EMI Level (dBm)')
    ax.set_title(
        'PCB Radiated EMI vs Compliance Limit',
        fontweight='bold'
    )
    ax.legend(loc='upper right')
    ax.set_xlim(0, max(80, predicted_emi + 15))
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)
    plt.close()

    st.markdown("---")
    st.subheader("📈 EMI Across Frequency Range")

    freqs    = np.linspace(30, 1000, 60)
    emi_vals = []
    for f in freqs:
        tmp = pcb_row.copy()
        tmp['frequency_mhz'] = float(f)
        emi_vals.append(
            calculate_emi(tmp, model, input_size)
        )

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(
        freqs, emi_vals,
        color='blue', linewidth=2,
        label='Predicted EMI'
    )
    ax2.axhline(
        y=LIMIT, color='red',
        linestyle='--', linewidth=2,
        label=f'Limit ({LIMIT} dBm)'
    )
    ax2.fill_between(
        freqs, emi_vals, LIMIT,
        where=[e > LIMIT for e in emi_vals],
        alpha=0.3, color='red',
        label='Violation Zone'
    )
    ax2.fill_between(
        freqs, emi_vals, LIMIT,
        where=[e <= LIMIT for e in emi_vals],
        alpha=0.2, color='green',
        label='Safe Zone'
    )
    ax2.set_xlabel('Frequency (MHz)')
    ax2.set_ylabel('Predicted EMI (dBm)')
    ax2.set_title(
        'EMI Spectrum — 30 MHz to 1000 MHz',
        fontweight='bold'
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)
    plt.close()

    st.markdown("---")
    st.subheader("📋 Current PCB Parameters")
    st.dataframe(
        pd.DataFrame({
            'Parameter': [
                'Trace Width (mm)',
                'Trace Length (mm)',
                'Ground Distance (mm)',
                'Stitching Vias',
                'Decap Distance (mm)',
                'Frequency (MHz)'
            ],
            'Value': [
                str(trace_width),
                str(trace_length),
                str(ground_distance),
                str(stitching_vias),
                str(decap_distance),
                str(frequency)
            ]
        }),
        hide_index=True,
        width='stretch'
    )

# ════════════════════════════════════════
# TAB 2: ROOT CAUSE
# ════════════════════════════════════════

with tab2:
    st.subheader("🔍 Root Cause Analysis")

    scores = {
        'Trace Length':
            trace_length / 100.0,
        'Frequency':
            frequency / 1000.0,
        'Decap Distance':
            decap_distance / 15.0,
        'Ground Distance':
            ground_distance / 2.0,
        'Fewer Vias':
            1.0 - stitching_vias / 10.0,
        'Thin Trace':
            1.0 - trace_width / 2.0,
    }

    fixes = {
        'Trace Length':
            'Shorten trace or add return path vias',
        'Frequency':
            'Add ferrite bead or LC filter on net',
        'Decap Distance':
            'Move decap within 0.5mm of IC pin',
        'Ground Distance':
            'Add ground copper pour',
        'Fewer Vias':
            'Add stitching vias every 5mm',
        'Thin Trace':
            'Increase trace width to 0.8mm+',
    }

    sorted_scores = sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    for rank, (param, score) in \
            enumerate(sorted_scores):
        color = (
            "🔴" if score > 0.7 else
            "🟡" if score > 0.4 else
            "🟢"
        )
        with st.expander(
            f"{color} #{rank+1} — {param} "
            f"(Impact: {score:.2f})",
            expanded=(rank < 3)
        ):
            st.progress(float(score))
            ca, cb = st.columns(2)
            with ca:
                st.info(
                    f"**Score:** {score:.3f}"
                )
            with cb:
                st.warning(
                    f"**Fix:** {fixes[param]}"
                )

    st.markdown("---")

    fig3, ax3 = plt.subplots(figsize=(9, 5))
    p_list = [s[0] for s in sorted_scores]
    v_list = [s[1] for s in sorted_scores]
    c_list = [
        'red'    if v > 0.7 else
        'orange' if v > 0.4 else
        'green'
        for v in v_list
    ]
    ax3.barh(
        p_list, v_list,
        color=c_list,
        edgecolor='black', alpha=0.8
    )
    ax3.set_xlabel('EMI Impact Score')
    ax3.set_title(
        'Root Cause Analysis',
        fontweight='bold'
    )
    ax3.axvline(
        x=0.5, color='black',
        linestyle='--', alpha=0.5
    )
    ax3.grid(True, alpha=0.3, axis='x')
    st.pyplot(fig3)
    plt.close()

    st.markdown("---")
    st.subheader("🔧 Top 3 Recommended Fixes")
    fix_data = []
    for rank, (param, score) in \
            enumerate(sorted_scores[:3]):
        fix_data.append({
            'Priority':  f"#{rank+1}",
            'Parameter': param,
            'Score':     f"{score:.3f}",
            'Fix':       fixes[param]
        })
    st.dataframe(
        pd.DataFrame(fix_data),
        hide_index=True,
        width='stretch'
    )

# ════════════════════════════════════════
# TAB 3: OPTIMIZER
# ════════════════════════════════════════

with tab3:
    st.subheader("⚡ AI-Powered Layout Optimizer")

    cb1, cb2 = st.columns([1, 3])
    with cb1:
        run_opt = st.button(
            "🚀 Run Optimizer",
            type="primary"
        )
    with cb2:
        iterations = st.select_slider(
            "Optimization Iterations",
            options=[500, 1000, 2000, 3000],
            value=2000
        )

    if run_opt:
        with st.spinner(
            f"Running {iterations} iterations..."
        ):
            best_pcb, best_emi, init_emi, history\
                = optimize_pcb(
                    pcb_row, model, input_size,
                    iterations=iterations
                )

        st.success("✅ Optimization Complete!")
        st.markdown("---")

        improvement = init_emi - best_emi
        cx, cy, cz  = st.columns(3)
        with cx:
            st.metric(
                "EMI Before",
                f"{init_emi:.2f} dBm"
            )
        with cy:
            st.metric(
                "EMI After",
                f"{best_emi:.2f} dBm",
                delta=f"-{improvement:.2f} dBm"
            )
        with cz:
            if best_emi < LIMIT:
                st.success("✅ NOW PASSES!")
            else:
                st.warning(
                    f"⚠️ "
                    f"{best_emi - LIMIT:.1f} dBm"
                    f" still over"
                )

        st.markdown("### 📋 What the AI Changed")
        changes = []
        for key in [
            'trace_width_mm',
            'trace_length_mm',
            'ground_distance_mm',
            'stitching_vias',
            'decap_distance_mm',
            'frequency_mhz'
        ]:
            before = float(pcb_row[key])
            after  = float(
                best_pcb.get(key, before)
            )
            diff   = after - before
            changes.append({
                'Parameter': key,
                'Before':    f"{before:.2f}",
                'After':     f"{after:.2f}",
                'Change':    f"{diff:+.2f}",
                'Status': (
                    "✓ Changed"
                    if abs(diff) > 0.01
                    else "— Same"
                )
            })
        st.dataframe(
            pd.DataFrame(changes),
            hide_index=True,
            width='stretch'
        )

        fig4, ax4 = plt.subplots(figsize=(10, 4))
        ax4.plot(
            history, color='blue',
            linewidth=1.5, label='Best EMI'
        )
        ax4.axhline(
            y=LIMIT, color='red',
            linestyle='--',
            label=f'Limit ({LIMIT} dBm)'
        )
        ax4.axhline(
            y=init_emi, color='orange',
            linestyle=':',
            label=f'Initial ({init_emi:.1f} dBm)'
        )
        ax4.fill_between(
            range(len(history)),
            history, LIMIT,
            where=[h > LIMIT for h in history],
            alpha=0.2, color='red'
        )
        ax4.fill_between(
            range(len(history)),
            history, LIMIT,
            where=[h <= LIMIT for h in history],
            alpha=0.2, color='green'
        )
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Best EMI (dBm)')
        ax4.set_title(
            'Optimization Progress',
            fontweight='bold'
        )
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        st.pyplot(fig4)
        plt.close()

        os.makedirs('outputs', exist_ok=True)
        result = {
            'initial_emi':    float(init_emi),
            'optimized_emi':  float(best_emi),
            'improvement_db': float(improvement),
            'passes': bool(best_emi < LIMIT),
            'optimized_pcb': {
                k: float(v)
                for k, v in best_pcb.items()
            }
        }
        with open(
            'outputs/optimized_pcb.json', 'w'
        ) as f:
            json.dump(result, f, indent=2)

        st.download_button(
            "⬇️ Download Optimized PCB (JSON)",
            data=json.dumps(result, indent=2),
            file_name="optimized_pcb.json",
            mime="application/json"
        )
    else:
        st.info("👆 Click Run Optimizer to start.")
        st.markdown("""
        **How the optimizer works:**
        - Physics-guided hill climbing
        - Wider trace → less EMI
        - Shorter trace → less EMI
        - More vias → less EMI
        - Closer decap → less EMI
        - Closer ground → less EMI
        """)

# ════════════════════════════════════════
# TAB 4: MULTI-AGENT
# ════════════════════════════════════════

with tab4:
    st.subheader("🤖 Multi-Agent DRL Optimizer")
    st.markdown(
        "4 specialist AI agents work as a team "
        "to find the best PCB layout!"
    )

    st.markdown("### 👥 Meet Your AI Team")
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        st.info(
            "**⚡ PDN Specialist**\n\n"
            "Optimizes power delivery\n"
            "Controls trace width\n"
            "and decap placement"
        )
    with a2:
        st.info(
            "**📡 SI Specialist**\n\n"
            "Optimizes signal integrity\n"
            "Controls trace length\n"
            "and impedance matching"
        )
    with a3:
        st.info(
            "**🔄 Return Path Agent**\n\n"
            "Optimizes ground return\n"
            "Controls stitching vias\n"
            "and ground distance"
        )
    with a4:
        st.info(
            "**🔋 Decoupling Agent**\n\n"
            "Optimizes capacitor placement\n"
            "Controls decap distance\n"
            "and SRF management"
        )

    st.markdown("---")

    ma1, ma2 = st.columns([1, 3])
    with ma1:
        run_multi = st.button(
            "🚀 Run Multi-Agent",
            type="primary"
        )
    with ma2:
        ma_iterations = st.select_slider(
            "Team Iterations",
            options=[500, 1000, 2000],
            value=1000,
            key='ma_iter'
        )

    ma_results_path = \
        'outputs/multi_agent_results.json'
    ma_chart_path   = \
        'outputs/multi_agent_results.png'

    if run_multi:
        with st.spinner(
            "All 4 agents optimizing... "
            "(3-4 minutes)"
        ):
            try:
                sys.path.insert(0, 'models')
                from multi_agent_optimizer import (
                    MultiAgentCoordinator
                )
                coordinator = MultiAgentCoordinator(
                    model, input_size
                )
                best_pcb, best_emi, history = \
                    coordinator.optimize(
                        pcb_row,
                        iterations=ma_iterations
                    )
                init_emi     = calculate_emi(
                    pcb_row, model, input_size
                )
                improvement  = init_emi - best_emi
                agent_report = \
                    coordinator.get_agent_report()

                st.success(
                    "✅ Multi-Agent Complete!"
                )
                st.markdown("---")
                st.markdown("### 📊 Team Results")

                mr1, mr2, mr3, mr4 = st.columns(4)
                with mr1:
                    st.metric(
                        "EMI Before",
                        f"{init_emi:.2f} dBm"
                    )
                with mr2:
                    st.metric(
                        "EMI After",
                        f"{best_emi:.2f} dBm",
                        delta=(
                            f"-{improvement:.2f}"
                            f" dBm"
                        )
                    )
                with mr3:
                    st.metric(
                        "Improvement",
                        f"{improvement:.2f} dBm"
                    )
                with mr4:
                    if best_emi < LIMIT:
                        st.success("✅ PASS!")
                    else:
                        st.warning("⚠️ Improved")

                names = list(agent_report.keys())
                wins  = [
                    agent_report[n]['wins']
                    for n in names
                ]
                short = [
                    'PDN', 'SI',
                    'Return', 'Decap'
                ]

                fig_ma, axes_ma = plt.subplots(
                    1, 2, figsize=(14, 5)
                )
                axes_ma[0].plot(
                    history, color='blue',
                    linewidth=1.5,
                    label='Best EMI'
                )
                axes_ma[0].axhline(
                    y=LIMIT, color='red',
                    linestyle='--',
                    label=f'Limit ({LIMIT} dBm)'
                )
                axes_ma[0].axhline(
                    y=init_emi, color='orange',
                    linestyle=':',
                    label=(
                        f'Start '
                        f'({init_emi:.1f} dBm)'
                    )
                )
                axes_ma[0].fill_between(
                    range(len(history)),
                    history, LIMIT,
                    where=[
                        h > LIMIT for h in history
                    ],
                    alpha=0.2, color='red'
                )
                axes_ma[0].fill_between(
                    range(len(history)),
                    history, LIMIT,
                    where=[
                        h <= LIMIT for h in history
                    ],
                    alpha=0.2, color='green'
                )
                axes_ma[0].set_xlabel('Iteration')
                axes_ma[0].set_ylabel('EMI (dBm)')
                axes_ma[0].set_title(
                    'Multi-Agent Progress'
                )
                axes_ma[0].legend()
                axes_ma[0].grid(True, alpha=0.3)

                if sum(wins) > 0:
                    axes_ma[1].pie(
                        wins,
                        labels=short,
                        autopct='%1.1f%%',
                        startangle=90,
                        pctdistance=0.75,
                        labeldistance=1.15,
                        colors=[
                            '#ff6b6b', '#4ecdc4',
                            '#45b7d1', '#96ceb4'
                        ]
                    )
                    axes_ma[1].set_title(
                        'Agent Contributions'
                    )

                plt.tight_layout()
                st.pyplot(fig_ma)
                plt.close()

                st.markdown("---")
                st.markdown(
                    "### 📋 What the Team Changed"
                )
                changes = []
                for key in [
                    'trace_width_mm',
                    'trace_length_mm',
                    'ground_distance_mm',
                    'stitching_vias',
                    'decap_distance_mm'
                ]:
                    before = float(pcb_row[key])
                    after  = float(
                        best_pcb.get(key, before)
                    )
                    diff   = after - before
                    changes.append({
                        'Parameter': key,
                        'Before': f"{before:.2f}",
                        'After':  f"{after:.2f}",
                        'Change': f"{diff:+.2f}"
                    })
                st.dataframe(
                    pd.DataFrame(changes),
                    hide_index=True,
                    width='stretch'
                )

                st.markdown("---")
                st.markdown(
                    "### 🏅 Agent Leaderboard"
                )
                board_data = []
                for name, data in \
                        agent_report.items():
                    board_data.append({
                        'Agent': name,
                        'Wins':  str(data['wins']),
                        'Contribution':
                            data['contribution']
                    })
                st.dataframe(
                    pd.DataFrame(
                        board_data
                    ).sort_values(
                        'Wins', ascending=False
                    ),
                    hide_index=True,
                    width='stretch'
                )

                os.makedirs(
                    'outputs', exist_ok=True
                )
                ma_result = {
                    'initial_emi':
                        float(init_emi),
                    'optimized_emi':
                        float(best_emi),
                    'improvement_db':
                        float(improvement),
                    'passes_cispr32':
                        bool(best_emi < LIMIT),
                    'optimized_pcb': {
                        k: float(v)
                        for k, v
                        in best_pcb.items()
                    },
                    'agent_report': agent_report
                }
                with open(
                    'outputs/'
                    'multi_agent_results.json',
                    'w'
                ) as f:
                    json.dump(
                        ma_result, f, indent=2
                    )

                st.download_button(
                    "⬇️ Download Team Results",
                    data=json.dumps(
                        ma_result, indent=2
                    ),
                    file_name=(
                        "multi_agent_results.json"
                    ),
                    mime="application/json"
                )

            except Exception as e:
                st.error(f"Error: {e}")

    elif os.path.exists(ma_results_path):
        st.info(
            "Showing previous results. "
            "Click Run for new run."
        )
        with open(ma_results_path) as f:
            prev = json.load(f)
        p1, p2, p3 = st.columns(3)
        with p1:
            st.metric(
                "EMI Before",
                f"{prev['initial_emi']:.2f} dBm"
            )
        with p2:
            st.metric(
                "EMI After",
                f"{prev['optimized_emi']:.2f} dBm"
            )
        with p3:
            if prev['passes_cispr32']:
                st.success("✅ PASSED!")
            else:
                st.warning("⚠️ Improved")
        if os.path.exists(ma_chart_path):
            st.image(
                ma_chart_path,
                caption="Previous Run Results",
                use_container_width=True
            )
    else:
        st.info(
            "👆 Click Run Multi-Agent to start!"
        )

# ════════════════════════════════════════
# TAB 5: REPORT
# ════════════════════════════════════════

with tab5:
    st.subheader("📋 Analysis Report")

    train_path  = \
        'outputs/real_training_results.png'
    report_path = 'outputs/final_report.png'
    json_path   = 'outputs/final_report.json'

    if os.path.exists(train_path):
        st.markdown("### 🧠 Training Results")
        st.image(
            train_path,
            caption="KAN-PINN Training Results",
            use_container_width=True
        )

    if os.path.exists(report_path):
        st.markdown("### 📊 Pipeline Report")
        st.image(
            report_path,
            caption="Full Analysis Report",
            use_container_width=True
        )
    else:
        st.info(
            "Run pipeline first:\n"
            "```\npython run_pipeline.py\n```"
        )

    if os.path.exists(json_path):
        st.markdown("### 📄 JSON Report")
        with open(json_path) as f:
            report_data = json.load(f)
        st.json(report_data)
        st.download_button(
            "⬇️ Download Report (JSON)",
            data=json.dumps(
                report_data, indent=2
            ),
            file_name="neuroshield_report.json",
            mime="application/json"
        )

    st.markdown("---")
    st.markdown("### 📊 Current Session")
    summary_data = [
        ['Trace Width (mm)',
            str(trace_width)],
        ['Trace Length (mm)',
            str(trace_length)],
        ['Ground Distance (mm)',
            str(ground_distance)],
        ['Stitching Vias',
            str(stitching_vias)],
        ['Decap Distance (mm)',
            str(decap_distance)],
        ['Frequency (MHz)',
            str(frequency)],
        ['Predicted EMI (dBm)',
            str(round(predicted_emi, 2))],
        ['Standard', standard],
        ['Status',
            'PASS' if predicted_emi <= LIMIT
            else 'FAIL']
    ]
    st.dataframe(
        pd.DataFrame(
            summary_data,
            columns=['Parameter', 'Value']
        ),
        hide_index=True,
        width='stretch'
    )
    st.download_button(
        "⬇️ Download Current Analysis (JSON)",
        data=json.dumps(
            {r[0]: r[1] for r in summary_data},
            indent=2
        ),
        file_name="current_analysis.json",
        mime="application/json"
    )

# ════════════════════════════════════════
# TAB 6: BATCH ANALYSIS
# ════════════════════════════════════════

with tab6:
    st.subheader("📂 Batch PCB Analysis")

    sample_csv = (
        "trace_width_mm,trace_length_mm,"
        "ground_distance_mm,stitching_vias,"
        "decap_distance_mm,frequency_mhz\n"
        "0.5,50,0.5,3,5.0,500\n"
        "0.2,90,1.5,1,12.0,800\n"
        "1.0,20,0.2,8,2.0,100\n"
        "0.3,75,1.0,2,10.0,600\n"
        "1.5,10,0.1,9,1.0,200"
    )
    st.download_button(
        "⬇️ Download Sample CSV Template",
        data=sample_csv,
        file_name="sample_pcb_designs.csv",
        mime="text/csv"
    )

    if uploaded_file is not None:
        df_up = pd.read_csv(uploaded_file)
        st.success(
            f"✅ Loaded {len(df_up)} designs!"
        )
        st.dataframe(
            df_up,
            hide_index=True,
            width='stretch'
        )

        if st.button("🔍 Analyze All Boards"):
            results  = []
            progress = st.progress(0)
            status   = st.empty()

            for i, row in df_up.iterrows():
                pcb = {
                    'trace_width_mm':
                        float(row.get(
                            'trace_width_mm', 0.5
                        )),
                    'trace_length_mm':
                        float(row.get(
                            'trace_length_mm', 50
                        )),
                    'ground_distance_mm':
                        float(row.get(
                            'ground_distance_mm',
                            0.5
                        )),
                    'stitching_vias':
                        float(row.get(
                            'stitching_vias', 3
                        )),
                    'decap_distance_mm':
                        float(row.get(
                            'decap_distance_mm', 5
                        )),
                    'frequency_mhz':
                        float(row.get(
                            'frequency_mhz', 500
                        )),
                }
                emi = calculate_emi(
                    pcb, model, input_size
                )
                results.append({
                    'Board #':
                        str(i + 1),
                    'EMI (dBm)':
                        f"{emi:.2f}",
                    'Status':
                        'PASS' if emi < LIMIT
                        else 'FAIL',
                    'Margin':
                        f"{abs(emi-LIMIT):.2f} dBm"
                })
                progress.progress(
                    (i + 1) / len(df_up)
                )
                status.text(
                    f"Board {i+1}/{len(df_up)}..."
                )

            progress.empty()
            status.empty()

            results_df = pd.DataFrame(results)
            st.dataframe(
                results_df,
                hide_index=True,
                width='stretch'
            )

            pass_count = sum(
                1 for r in results
                if r['Status'] == 'PASS'
            )
            fail_count = len(results) - pass_count
            total      = len(results)

            r1, r2, r3 = st.columns(3)
            with r1:
                st.metric("Total", total)
            with r2:
                st.metric(
                    "Passing",
                    f"{pass_count} "
                    f"({pass_count*100//total}%)"
                )
            with r3:
                st.metric(
                    "Failing",
                    f"{fail_count} "
                    f"({fail_count*100//total}%)"
                )

            st.download_button(
                "⬇️ Download Results (CSV)",
                data=results_df.to_csv(
                    index=False
                ),
                file_name="batch_results.csv",
                mime="text/csv"
            )
    else:
        st.info(
            "👆 Upload CSV file in the sidebar."
        )

# ════════════════════════════════════════
# TAB 7: KICAD ANALYSIS
# ════════════════════════════════════════

with tab7:
    st.subheader("🔌 KiCad PCB File Analyzer")
    st.markdown(
        "Upload a real KiCad PCB file and get "
        "instant EMI analysis for every trace!"
    )

    kicad_file = st.file_uploader(
        "Upload KiCad PCB File (.kicad_pcb)",
        type=['kicad_pcb']
    )
    freq_kicad = st.slider(
        "Analysis Frequency (MHz)",
        30, 1000, 500, 10,
        key='kicad_freq'
    )

    def run_kicad_analysis(filepath, freq):
        from kicad_reader import (
            KiCadPCBReader,
            PCBEMIAnalyzer
        )
        reader   = KiCadPCBReader(filepath)
        reader.read()
        analyzer = PCBEMIAnalyzer(
            reader, model, input_size, freq
        )
        results  = analyzer.analyze_all_traces()
        board    = analyzer.get_board_summary()
        return reader, results, board

    def show_kicad_results(results, board):
        st.markdown("---")
        st.subheader("📊 Board Summary")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric(
                "Total Traces",
                str(board['total_traces'])
            )
        with c2:
            st.metric(
                "Max EMI",
                f"{board['max_emi']} dBm"
            )
        with c3:
            st.metric(
                "High Risk",
                f"{board['high_risk']} traces"
            )
        with c4:
            if board['board_status'] == 'PASS':
                st.success("✅ PASS")
            else:
                st.error("❌ FAIL")

        st.markdown("---")
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            st.error(
                f"🔴 High Risk\n\n"
                f"**{board['high_risk']}** traces"
            )
        with rc2:
            st.warning(
                f"🟡 Medium Risk\n\n"
                f"**{board['medium_risk']}** traces"
            )
        with rc3:
            st.success(
                f"🟢 Low Risk\n\n"
                f"**{board['low_risk']}** traces"
            )

        st.markdown("---")
        st.dataframe(
            pd.DataFrame([{
                'Trace':  str(r['trace_id']),
                'Net':    str(r['net_name']),
                'Layer':  str(r['layer']),
                'Width':  f"{r['width_mm']:.2f}mm",
                'Length': f"{r['length_mm']:.1f}mm",
                'EMI':    f"{r['emi_dbm']:.1f} dBm",
                'Risk':   str(r['risk']),
                'Status': str(r['status'])
            } for r in results]),
            hide_index=True,
            width='stretch'
        )

        fig_k, ax_k = plt.subplots(
            figsize=(10, 4)
        )
        trace_ids = [
            f"T{r['trace_id']}\n{r['net_name']}"
            for r in results
        ]
        emi_vals = [r['emi_dbm'] for r in results]
        colors_k = [
            'red'    if r['risk'] == 'HIGH' else
            'orange' if r['risk'] == 'MEDIUM' else
            'green'
            for r in results
        ]
        ax_k.bar(
            trace_ids, emi_vals,
            color=colors_k,
            edgecolor='black', alpha=0.8
        )
        ax_k.axhline(
            y=40.0, color='red',
            linestyle='--', linewidth=2,
            label='CISPR 32 Limit (40 dBm)'
        )
        ax_k.set_xlabel('Trace')
        ax_k.set_ylabel('EMI (dBm)')
        ax_k.set_title(
            'EMI Risk per Trace',
            fontweight='bold'
        )
        ax_k.legend()
        ax_k.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        st.pyplot(fig_k)
        plt.close()

        st.download_button(
            "⬇️ Download Analysis (JSON)",
            data=json.dumps({
                'board_summary': board,
                'trace_results': results
            }, indent=2),
            file_name="kicad_analysis.json",
            mime="application/json"
        )

    if kicad_file is not None:
        temp_path = \
            'data/uploaded_board.kicad_pcb'
        with open(temp_path, 'w') as f:
            f.write(
                kicad_file.read().decode('utf-8')
            )
        st.success(
            f"✅ Uploaded: {kicad_file.name}"
        )
        if st.button(
            "🔍 Analyze PCB",
            type="primary"
        ):
            with st.spinner("Analyzing EMI..."):
                try:
                    reader, results, board = \
                        run_kicad_analysis(
                            temp_path, freq_kicad
                        )
                    show_kicad_results(
                        results, board
                    )
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.info(
            "👆 Upload a .kicad_pcb file above."
        )
        st.markdown("---")
        st.markdown("### 🧪 Try Sample Board")
        if st.button("📋 Analyze Sample Board"):
            with st.spinner("Analyzing..."):
                try:
                    reader, results, board = \
                        run_kicad_analysis(
                            'data/sample_board'
                            '.kicad_pcb',
                            freq_kicad
                        )
                    st.success("✅ Done!")
                    show_kicad_results(
                        results, board
                    )
                except Exception as e:
                    st.error(f"Error: {e}")

# ════════════════════════════════════════
# TAB 8: FEDERATED LEARNING
# ════════════════════════════════════════

with tab8:
    st.subheader("🌐 Federated Learning System")
    st.markdown(
        "Train AI across multiple companies "
        "without sharing private PCB designs!"
    )

    st.markdown("### 🏭 How It Works")
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        st.info(
            "**Step 1**\n\n"
            "Each company trains on\n"
            "their PRIVATE boards\n"
            "locally on their server"
        )
    with fc2:
        st.info(
            "**Step 2**\n\n"
            "Only model WEIGHTS\n"
            "are sent to server\n"
            "Never actual designs!"
        )
    with fc3:
        st.info(
            "**Step 3**\n\n"
            "Server combines weights\n"
            "using FedAvg algorithm\n"
            "Weighted by dataset size"
        )
    with fc4:
        st.info(
            "**Step 4**\n\n"
            "Everyone receives\n"
            "smarter global model\n"
            "Privacy preserved!"
        )

    st.markdown("---")
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        st.success(
            "**🏠 Company A**\n\n"
            "Consumer Electronics\n"
            "30-300 MHz\n"
            "200 private designs"
        )
    with cc2:
        st.success(
            "**🏭 Company B**\n\n"
            "Industrial Equipment\n"
            "200-600 MHz\n"
            "200 private designs"
        )
    with cc3:
        st.success(
            "**🚗 Company C**\n\n"
            "Automotive Electronics\n"
            "500-1000 MHz\n"
            "200 private designs"
        )

    st.markdown("---")
    fl1, fl2, fl3 = st.columns(3)
    with fl1:
        run_fed = st.button(
            "🚀 Run Federated Training",
            type="primary"
        )
    with fl2:
        num_rounds = st.select_slider(
            "Training Rounds",
            options=[5, 10, 15, 20],
            value=10,
            key='fed_rounds'
        )
    with fl3:
        local_epochs = st.select_slider(
            "Local Epochs per Round",
            options=[3, 5, 10],
            value=5,
            key='fed_epochs'
        )

    fed_results_path = \
        'outputs/federated_report.json'
    fed_chart_path   = \
        'outputs/federated_results.png'

    if run_fed:
        with st.spinner(
            "3 companies training together... "
            "(3-5 minutes)"
        ):
            try:
                sys.path.insert(0, 'models')
                sys.path.insert(0, 'data')
                from federated_learning import (
                    generate_company_data,
                    FederatedClient,
                    FederatedServer,
                    generate_privacy_report
                )

                data_a = generate_company_data(
                    'Company_A',
                    num_samples=200, seed=42
                )
                data_b = generate_company_data(
                    'Company_B',
                    num_samples=200, seed=123
                )
                data_c = generate_company_data(
                    'Company_C',
                    num_samples=200, seed=456
                )

                test_row = {
                    'trace_width_mm':     0.5,
                    'trace_length_mm':    50.0,
                    'ground_distance_mm': 0.5,
                    'stitching_vias':     3.0,
                    'decap_distance_mm':  5.0,
                    'frequency_mhz':      500.0,
                }
                G         = build_pcb_graph(
                    pd.Series(test_row)
                )
                fv        = \
                    graph_to_feature_vector(G)
                fed_input = len(fv)

                client_a = FederatedClient(
                    'Company_A', data_a,
                    fed_input, local_epochs
                )
                client_b = FederatedClient(
                    'Company_B', data_b,
                    fed_input, local_epochs
                )
                client_c = FederatedClient(
                    'Company_C', data_c,
                    fed_input, local_epochs
                )

                server = FederatedServer(
                    input_size=fed_input,
                    num_rounds=num_rounds
                )
                server.add_client(client_a)
                server.add_client(client_b)
                server.add_client(client_c)
                server.train()
                server.save_global_model(
                    'outputs/federated_model.pth'
                )

                privacy_report = \
                    generate_privacy_report(
                        [client_a,
                         client_b,
                         client_c],
                        server
                    )
                with open(
                    'outputs/federated_report.json',
                    'w'
                ) as f:
                    json.dump(
                        privacy_report, f,
                        indent=2
                    )

                fig_fed = server.plot_results()
                st.success(
                    "✅ Federated Training Complete!"
                )
                st.markdown("---")
                st.markdown(
                    "### 📊 Training Results"
                )

                h = server.round_history
                fm1, fm2, fm3, fm4 = \
                    st.columns(4)
                with fm1:
                    st.metric("Companies", "3")
                with fm2:
                    st.metric(
                        "Rounds",
                        str(num_rounds)
                    )
                with fm3:
                    st.metric(
                        "Initial MAE",
                        f"{h[0]['avg_mae']:.2f} dBm"
                    )
                with fm4:
                    impr = (
                        h[0]['avg_mae'] -
                        h[-1]['avg_mae']
                    )
                    st.metric(
                        "Final MAE",
                        f"{h[-1]['avg_mae']:.2f} dBm",
                        delta=f"-{impr:.2f} dBm"
                    )

                st.pyplot(fig_fed)
                plt.close()

                st.markdown("---")
                st.markdown(
                    "### 📋 Round by Round"
                )
                round_data = []
                for r in h:
                    round_data.append({
                        'Round':
                            str(r['round']),
                        'Avg Loss':
                            f"{r['avg_loss']:.4f}",
                        'Global MAE (dBm)':
                            f"{r['avg_mae']:.4f}"
                    })
                st.dataframe(
                    pd.DataFrame(round_data),
                    hide_index=True,
                    width='stretch'
                )

                st.markdown("---")
                pv1, pv2, pv3 = st.columns(3)
                with pv1:
                    st.error(
                        "🚫 **Board Designs**\n\n"
                        "NEVER shared"
                    )
                with pv2:
                    st.error(
                        "🚫 **EMI Values**\n\n"
                        "NEVER shared"
                    )
                with pv3:
                    st.success(
                        "✅ **Weights Only**\n\n"
                        "Anonymous numbers"
                    )

                st.download_button(
                    "⬇️ Download Privacy Report",
                    data=json.dumps(
                        privacy_report, indent=2
                    ),
                    file_name=(
                        "federated_report.json"
                    ),
                    mime="application/json"
                )

            except Exception as e:
                st.error(f"Error: {e}")

    elif os.path.exists(fed_results_path):
        st.info("Showing previous results.")
        with open(fed_results_path) as f:
            prev_fed = json.load(f)

        pf1, pf2, pf3 = st.columns(3)
        with pf1:
            st.metric(
                "Companies",
                str(len(
                    prev_fed.get('clients', {})
                ))
            )
        with pf2:
            final_mae = prev_fed.get(
                'global_model', {}
            ).get('final_mae', 0)
            st.metric(
                "Final MAE",
                f"{final_mae:.4f} dBm"
            )
        with pf3:
            impr = prev_fed.get(
                'global_model', {}
            ).get('improvement', 0)
            st.metric(
                "Improvement",
                f"{impr:.4f} dBm"
            )

        if os.path.exists(fed_chart_path):
            st.image(
                fed_chart_path,
                caption=(
                    "Federated Learning Results"
                ),
                use_container_width=True
            )

        pv1, pv2, pv3 = st.columns(3)
        with pv1:
            st.error(
                "🚫 **Board Designs**\n\n"
                "NEVER shared"
            )
        with pv2:
            st.error(
                "🚫 **EMI Values**\n\n"
                "NEVER shared"
            )
        with pv3:
            st.success(
                "✅ **Weights Only**\n\n"
                "Anonymous numbers"
            )

    else:
        st.info(
            "👆 Click Run Federated Training!"
        )
        ra1, ra2, ra3 = st.columns(3)
        with ra1:
            st.info(
                "**📱 Apple**\n\n"
                "Trains keyboard prediction\n"
                "without reading your texts"
            )
        with ra2:
            st.info(
                "**🏥 Hospitals**\n\n"
                "Train cancer AI together\n"
                "without sharing patient data"
            )
        with ra3:
            st.info(
                "**🚗 Car Companies**\n\n"
                "Train self-driving AI\n"
                "without sharing road data"
            )

# ════════════════════════════════════════
# TAB 9: DIGITAL TWIN
# ════════════════════════════════════════

with tab9:
    st.subheader("🔬 Digital Twin System")
    st.markdown(
        "Real-time AI that learns from actual "
        "EMC measurements — gets smarter "
        "every test cycle automatically!"
    )

    st.markdown("### 🧠 How Digital Twin Works")
    dt1, dt2, dt3, dt4 = st.columns(4)
    with dt1:
        st.info(
            "**Step 1: Predict**\n\n"
            "AI predicts EMI\n"
            "for your PCB design\n"
            "before testing"
        )
    with dt2:
        st.info(
            "**Step 2: Measure**\n\n"
            "Real EMC lab measures\n"
            "actual board EMI\n"
            "in test chamber"
        )
    with dt3:
        st.warning(
            "**Step 3: Compare**\n\n"
            "System compares\n"
            "prediction vs reality\n"
            "detects any drift"
        )
    with dt4:
        st.success(
            "**Step 4: Update**\n\n"
            "AI auto-corrects\n"
            "itself from new data\n"
            "gets smarter!"
        )

    st.markdown("---")

    dc1, dc2, dc3 = st.columns(3)
    with dc1:
        run_twin = st.button(
            "🚀 Run Digital Twin",
            type="primary"
        )
    with dc2:
        num_cycles = st.select_slider(
            "Measurement Cycles",
            options=[20, 50, 100],
            value=50,
            key='dt_cycles'
        )
    with dc3:
        noise_level = st.select_slider(
            "Lab Noise Level (dBm)",
            options=[0.5, 1.0, 1.5, 2.0, 3.0],
            value=1.5,
            key='dt_noise'
        )

    dt_results_path = \
        'outputs/digital_twin_results.json'
    dt_chart_path   = \
        'outputs/digital_twin_results.png'

    if run_twin:
        with st.spinner(
            f"Running {num_cycles} measurement "
            "cycles... (1-2 minutes)"
        ):
            try:
                sys.path.insert(0, 'models')
                from digital_twin import (
                    DigitalTwinEngine
                )

                twin = DigitalTwinEngine(
                    model, input_size,
                    noise_std=noise_level
                )
                results = twin.run_simulation(
                    num_cycles=num_cycles,
                    vary_params=True
                )
                summary = twin.get_summary()
                fig_dt  = twin.plot_results()

                st.success(
                    "✅ Digital Twin Complete!"
                )
                st.markdown("---")

                health = summary['final_health']
                if health >= 80:
                    st.success(
                        f"🟢 Model Health: "
                        f"{health}% — EXCELLENT"
                    )
                elif health >= 60:
                    st.warning(
                        f"🟡 Model Health: "
                        f"{health}% — GOOD"
                    )
                else:
                    st.error(
                        f"🔴 Model Health: "
                        f"{health}% — NEEDS ATTENTION"
                    )

                st.markdown(
                    "### 📊 Session Summary"
                )
                sm1, sm2, sm3, sm4 = \
                    st.columns(4)
                with sm1:
                    st.metric(
                        "Total Cycles",
                        str(summary['total_cycles'])
                    )
                with sm2:
                    st.metric(
                        "Avg Error",
                        f"{summary['avg_error']} dBm"
                    )
                with sm3:
                    st.metric(
                        "Auto Updates",
                        str(summary['auto_updates'])
                    )
                with sm4:
                    st.metric(
                        "Final Health",
                        f"{summary['final_health']}%"
                    )

                st.markdown("---")
                sb1, sb2, sb3 = st.columns(3)
                with sb1:
                    st.success(
                        f"🟢 Healthy\n\n"
                        f"**{summary['healthy_cycles']}"
                        f"** cycles"
                    )
                with sb2:
                    st.warning(
                        f"🟡 Warning\n\n"
                        f"**{summary['warning_cycles']}"
                        f"** cycles"
                    )
                with sb3:
                    st.error(
                        f"🔴 Critical\n\n"
                        f"**{summary['critical_cycles']}"
                        f"** cycles"
                    )

                st.markdown("---")
                st.subheader(
                    "📈 Digital Twin Charts"
                )
                st.pyplot(fig_dt)
                plt.close()

                st.markdown("---")
                st.subheader("📋 Measurement Log")
                table_data = []
                for r in results[-20:]:
                    table_data.append({
                        'Cycle':
                            str(r['cycle']),
                        'Predicted (dBm)':
                            str(r['predicted']),
                        'Measured (dBm)':
                            str(r['measured']),
                        'Error (dBm)':
                            str(r['error']),
                        'Status':
                            r['status'],
                        'Auto-Updated':
                            "✓" if r['updated']
                            else "—"
                    })
                st.dataframe(
                    pd.DataFrame(table_data),
                    hide_index=True,
                    width='stretch'
                )
                st.caption(
                    "Showing last 20 cycles"
                )

                os.makedirs(
                    'outputs', exist_ok=True
                )
                with open(
                    'outputs/'
                    'digital_twin_results.json',
                    'w'
                ) as f:
                    json.dump({
                        'summary': summary,
                        'cycles':  results
                    }, f, indent=2)

                st.download_button(
                    "⬇️ Download Results (JSON)",
                    data=json.dumps({
                        'summary': summary,
                        'cycles':  results
                    }, indent=2),
                    file_name=(
                        "digital_twin_results.json"
                    ),
                    mime="application/json"
                )

            except Exception as e:
                st.error(f"Error: {e}")

    elif os.path.exists(dt_results_path):
        st.info(
            "Showing previous results. "
            "Click Run Digital Twin for new run."
        )
        with open(dt_results_path) as f:
            prev_dt = json.load(f)

        prev_summary = prev_dt.get('summary', {})
        ps1, ps2, ps3, ps4 = st.columns(4)
        with ps1:
            st.metric(
                "Total Cycles",
                str(prev_summary.get(
                    'total_cycles', 0
                ))
            )
        with ps2:
            st.metric(
                "Avg Error",
                f"{prev_summary.get('avg_error', 0)}"
                f" dBm"
            )
        with ps3:
            st.metric(
                "Auto Updates",
                str(prev_summary.get(
                    'auto_updates', 0
                ))
            )
        with ps4:
            st.metric(
                "Final Health",
                f"{prev_summary.get('final_health', 0)}"
                f"%"
            )

        if os.path.exists(dt_chart_path):
            st.image(
                dt_chart_path,
                caption="Digital Twin Results",
                use_container_width=True
            )

    else:
        st.info(
            "👆 Click Run Digital Twin to start!"
        )
        rw1, rw2, rw3 = st.columns(3)
        with rw1:
            st.info(
                "**🚗 Tesla**\n\n"
                "Digital twin of every car\n"
                "Updates from real driving\n"
                "Predicts failures early"
            )
        with rw2:
            st.info(
                "**✈️ Boeing**\n\n"
                "Digital twin of engines\n"
                "Real sensor data updates AI\n"
                "Prevents failures"
            )
        with rw3:
            st.info(
                "**🏭 Siemens**\n\n"
                "Digital twin of factories\n"
                "Optimizes production live\n"
                "Saves millions per year"
            )
# ════════════════════════════════════════
# TAB 10: ADVANCED AI (Phase 7)
# ════════════════════════════════════════

with tab10:
    st.subheader("🧠 Advanced AI — Phase 7")
    st.markdown(
        "Three cutting-edge AI upgrades: "
        "Uncertainty Quantification + "
        "Fourier Neural Operator + "
        "Generative Layout Designer"
    )

    # Sub-tabs for 3 features
    (sub1, sub2, sub3) = st.tabs([
        "📊 Uncertainty",
        "📡 Fourier Operator",
        "🎨 Generative Designer"
    ])

    # ── SUB TAB 1: UNCERTAINTY ──
    with sub1:
        st.subheader(
            "📊 Uncertainty Quantification"
        )
        st.markdown(
            "AI predicts EMI **and** tells you "
            "how confident it is — with "
            "probability of passing compliance!"
        )

        uc1, uc2, uc3 = st.columns(3)
        with uc1:
            st.info(
                "**🎯 What it does**\n\n"
                "Runs model 100 times\n"
                "with dropout ON\n"
                "Gets distribution of predictions"
            )
        with uc2:
            st.info(
                "**📐 Output**\n\n"
                "Mean ± std dBm\n"
                "95% confidence interval\n"
                "Probability of PASS/FAIL"
            )
        with uc3:
            st.info(
                "**🔬 Method**\n\n"
                "Monte Carlo Dropout\n"
                "Bayesian approximation\n"
                "Used in medical AI"
            )

        st.markdown("---")

        n_samples_uc = st.select_slider(
            "MC Samples",
            options=[50, 100, 200],
            value=100,
            key='uc_samples'
        )

        run_uc = st.button(
            "🚀 Run Uncertainty Analysis",
            type="primary",
            key='run_uc'
        )

        uc_path    = \
            'outputs/uncertainty_results.json'
        uc_img_path = \
            'outputs/uncertainty_results.png'

        if run_uc:
            with st.spinner(
                "Running MC Dropout... "
                "(1-2 minutes)"
            ):
                try:
                    sys.path.insert(0, 'models')
                    from uncertainty import (
                        MCDropoutPredictor,
                        ComplianceRiskAnalyzer,
                        plot_uncertainty_results
                    )

                    predictor = MCDropoutPredictor(
                        model, input_size,
                        n_samples=n_samples_uc
                    )

                    result = \
                        predictor\
                        .predict_with_uncertainty(
                            pcb_row
                        )
                    risk = ComplianceRiskAnalyzer(
                        limit_dbm=LIMIT
                    ).calculate_risk(result)

                    sweep = \
                        predictor\
                        .predict_frequency_sweep(
                            pcb_row,
                            n_freqs=20
                        )
                    sensitivity = \
                        predictor\
                        .analyze_parameter_sensitivity(
                            pcb_row
                        )

                    st.success(
                        "✅ Uncertainty Analysis Done!"
                    )
                    st.markdown("---")

                    # Metrics
                    um1, um2, um3, um4 = \
                        st.columns(4)
                    with um1:
                        st.metric(
                            "Mean EMI",
                            f"{result['mean']:.2f}"
                            f" dBm"
                        )
                    with um2:
                        st.metric(
                            "Uncertainty",
                            f"±{result['std']:.2f}"
                            f" dBm"
                        )
                    with um3:
                        st.metric(
                            "Confidence",
                            result['confidence']
                        )
                    with um4:
                        st.metric(
                            "Pass Probability",
                            f"{risk['prob_pass']*100:.1f}%"
                        )

                    # Risk banner
                    rl = risk['risk_level']
                    if rl in ['VERY LOW', 'LOW']:
                        st.success(
                            f"🟢 Risk Level: {rl}"
                        )
                    elif rl == 'MEDIUM':
                        st.warning(
                            f"🟡 Risk Level: {rl}"
                        )
                    else:
                        st.error(
                            f"🔴 Risk Level: {rl}"
                        )

                    # 95% CI
                    st.info(
                        f"**95% Confidence Interval:** "
                        f"[{result['ci_95_lo']:.1f}, "
                        f"{result['ci_95_hi']:.1f}] dBm"
                        f" — AI is 95% sure your "
                        f"PCB EMI is in this range!"
                    )

                    # Chart
                    os.makedirs(
                        'outputs', exist_ok=True
                    )
                    fig_uc = \
                        plot_uncertainty_results(
                            result, risk,
                            sweep, sensitivity
                        )
                    st.pyplot(fig_uc)
                    plt.close()

                    # Save
                    with open(uc_path, 'w') as f:
                        json.dump({
                            'prediction': result,
                            'risk':       risk
                        }, f, indent=2)

                    st.download_button(
                        "⬇️ Download Results",
                        data=json.dumps({
                            'prediction': result,
                            'risk':       risk
                        }, indent=2),
                        file_name=(
                            "uncertainty_results"
                            ".json"
                        ),
                        mime="application/json"
                    )

                except Exception as e:
                    st.error(f"Error: {e}")

        elif os.path.exists(uc_path):
            st.info(
                "Showing previous results. "
                "Click Run for fresh analysis."
            )
            with open(uc_path) as f:
                prev_uc = json.load(f)
            pred = prev_uc.get('prediction', {})
            risk = prev_uc.get('risk', {})

            pm1, pm2, pm3, pm4 = st.columns(4)
            with pm1:
                st.metric(
                    "Mean EMI",
                    f"{pred.get('mean', 0):.2f} dBm"
                )
            with pm2:
                st.metric(
                    "Uncertainty",
                    f"±{pred.get('std', 0):.2f} dBm"
                )
            with pm3:
                st.metric(
                    "Confidence",
                    pred.get('confidence', 'N/A')
                )
            with pm4:
                st.metric(
                    "Pass Probability",
                    f"{risk.get('prob_pass', 0)*100:.1f}%"
                )
            if os.path.exists(uc_img_path):
                st.image(
                    uc_img_path,
                    caption=(
                        "Uncertainty Analysis"
                    ),
                    use_container_width=True
                )
        else:
            st.info(
                "👆 Click Run Uncertainty "
                "Analysis to start!"
            )

    # ── SUB TAB 2: FOURIER OPERATOR ──
    with sub2:
        st.subheader(
            "📡 Fourier Neural Operator"
        )
        st.markdown(
            "Predicts **entire EMI spectrum** "
            "in one shot — 64 frequencies "
            "simultaneously!"
        )

        fo1, fo2, fo3 = st.columns(3)
        with fo1:
            st.info(
                "**⚡ Speed**\n\n"
                "One forward pass\n"
                "predicts 64 frequencies\n"
                "vs 64 separate calls"
            )
        with fo2:
            st.info(
                "**🧮 Method**\n\n"
                "FFT → learned weights\n"
                "→ IFFT (spectral conv)\n"
                "O(N log N) complexity"
            )
        with fo3:
            st.info(
                "**🌍 Used in**\n\n"
                "Weather forecasting\n"
                "Fluid dynamics\n"
                "EM field prediction"
            )

        st.markdown("---")

        fno_model_path = 'outputs/fno_model.pth'
        fno_img_path   = 'outputs/fno_results.png'

        run_fno = st.button(
            "🚀 Run FNO Prediction",
            type="primary",
            key='run_fno'
        )

        if run_fno:
            with st.spinner(
                "Running Fourier Neural Operator..."
            ):
                try:
                    sys.path.insert(0, 'models')
                    from fourier_operator import (
                        PCBFNO,
                        FNOPredictor
                    )

                    if os.path.exists(
                        fno_model_path
                    ):
                        ckpt = torch.load(
                            fno_model_path,
                            weights_only=False
                        )
                        fno = PCBFNO(
                            param_dim=6,
                            n_freqs=ckpt['n_freqs'],
                            channels=32,
                            modes=16,
                            n_layers=4
                        )
                        fno.load_state_dict(
                            ckpt['model_state_dict']
                        )
                        fno_pred = FNOPredictor(
                            fno,
                            ckpt['X_mean'],
                            ckpt['X_std'],
                            ckpt['Y_mean'],
                            ckpt['Y_std'],
                            np.array(ckpt['freqs'])
                        )
                        pred_freqs, pred_emi = \
                            fno_pred\
                            .predict_spectrum(
                                pcb_row
                            )

                        st.success(
                            "✅ FNO Prediction Done!"
                        )
                        st.markdown("---")

                        fn1, fn2, fn3 = \
                            st.columns(3)
                        with fn1:
                            st.metric(
                                "Frequencies",
                                str(len(pred_freqs))
                            )
                        with fn2:
                            st.metric(
                                "Peak EMI",
                                f"{max(pred_emi):.2f}"
                                f" dBm"
                            )
                        with fn3:
                            violations = sum(
                                1 for e in pred_emi
                                if e > LIMIT
                            )
                            st.metric(
                                "Violations",
                                f"{violations} "
                                f"frequencies"
                            )

                        # Chart
                        fig_fno, ax_fno = \
                            plt.subplots(
                                figsize=(10, 4)
                            )
                        ax_fno.plot(
                            pred_freqs, pred_emi,
                            color='blue',
                            linewidth=2,
                            label='FNO Prediction'
                        )
                        ax_fno.axhline(
                            y=LIMIT, color='red',
                            linestyle='--',
                            linewidth=2,
                            label=f'Limit ({LIMIT} dBm)'
                        )
                        ax_fno.fill_between(
                            pred_freqs,
                            pred_emi, LIMIT,
                            where=[
                                e > LIMIT
                                for e in pred_emi
                            ],
                            alpha=0.3, color='red',
                            label='Violation Zone'
                        )
                        ax_fno.fill_between(
                            pred_freqs,
                            pred_emi, LIMIT,
                            where=[
                                e <= LIMIT
                                for e in pred_emi
                            ],
                            alpha=0.2, color='green',
                            label='Safe Zone'
                        )
                        ax_fno.set_xlabel(
                            'Frequency (MHz)'
                        )
                        ax_fno.set_ylabel(
                            'EMI (dBm)'
                        )
                        ax_fno.set_title(
                            'Full EMI Spectrum '
                            '— One Forward Pass!',
                            fontweight='bold'
                        )
                        ax_fno.legend()
                        ax_fno.grid(
                            True, alpha=0.3
                        )
                        st.pyplot(fig_fno)
                        plt.close()

                        # Spectrum table
                        st.markdown(
                            "### 📋 Spectrum Data"
                        )
                        spec_data = []
                        step = max(
                            1, len(pred_freqs)//10
                        )
                        for i in range(
                            0, len(pred_freqs),
                            step
                        ):
                            spec_data.append({
                                'Frequency (MHz)':
                                    f"{pred_freqs[i]:.0f}",
                                'EMI (dBm)':
                                    f"{pred_emi[i]:.2f}",
                                'Status':
                                    'FAIL'
                                    if pred_emi[i] > LIMIT
                                    else 'PASS'
                            })
                        st.dataframe(
                            pd.DataFrame(
                                spec_data
                            ),
                            hide_index=True,
                            width='stretch'
                        )

                    else:
                        st.warning(
                            "FNO model not found. "
                            "Run python models/"
                            "fourier_operator.py first!"
                        )

                except Exception as e:
                    st.error(f"Error: {e}")

        elif os.path.exists(fno_img_path):
            st.info(
                "Showing previous FNO results."
            )
            st.image(
                fno_img_path,
                caption="FNO Training Results",
                use_container_width=True
            )
            if os.path.exists(fno_model_path):
                st.success(
                    "✅ FNO model ready! "
                    "Click Run FNO Prediction."
                )
        else:
            st.warning(
                "⚠️ Run this first:\n"
                "```\n"
                "python models/fourier_operator.py"
                "\n```"
            )

    # ── SUB TAB 3: GENERATIVE DESIGNER ──
    with sub3:
        st.subheader(
            "🎨 Generative Layout Designer"
        )
        st.markdown(
            "AI generates **brand new PCB layouts** "
            "predicted to pass EMC compliance — "
            "no manual design needed!"
        )

        gd1, gd2, gd3 = st.columns(3)
        with gd1:
            st.info(
                "**🧬 How it works**\n\n"
                "VAE learns the space\n"
                "of good PCB designs\n"
                "Samples new ones!"
            )
        with gd2:
            st.info(
                "**📐 Output**\n\n"
                "New PCB parameters\n"
                "Predicted to pass EMC\n"
                "Ranked by margin"
            )
        with gd3:
            st.info(
                "**🌍 Used in**\n\n"
                "Drug discovery\n"
                "Chip design\n"
                "Material science"
            )

        st.markdown("---")

        vae_path = 'outputs/vae_model.pth'
        gen_path = 'outputs/generated_designs.json'
        gen_img  = 'outputs/generative_results.png'

        gc1, gc2 = st.columns(2)
        with gc1:
            n_gen = st.select_slider(
                "Designs to Generate",
                options=[200, 500, 1000],
                value=500,
                key='n_gen'
            )
        with gc2:
            target_margin = st.select_slider(
                "Target Margin (dBm below limit)",
                options=[2.0, 3.0, 5.0, 8.0],
                value=3.0,
                key='tgt_margin'
            )

        run_gen = st.button(
            "🚀 Generate Compliant Designs",
            type="primary",
            key='run_gen'
        )

        if run_gen:
            with st.spinner(
                f"Generating {n_gen} designs... "
                "(2-3 minutes)"
            ):
                try:
                    sys.path.insert(0, 'models')
                    from generative_designer import (
                        PCBVAE,
                        GenerativeDesigner
                    )

                    if os.path.exists(vae_path):
                        ckpt = torch.load(
                            vae_path,
                            weights_only=False
                        )
                        vae = PCBVAE(
                            input_dim=6,
                            hidden_dim=64,
                            latent_dim=ckpt[
                                'latent_dim'
                            ]
                        )
                        vae.load_state_dict(
                            ckpt['vae_state_dict']
                        )

                        designer = \
                            GenerativeDesigner(
                                vae, model,
                                input_size,
                                ckpt['bounds'],
                                ckpt['params']
                            )

                        passing, all_designs = \
                            designer\
                            .generate_compliant_designs(
                                n_generate=n_gen,
                                target_margin=
                                    target_margin
                            )

                        st.success(
                            f"✅ Generated "
                            f"{len(passing)} "
                            f"compliant designs!"
                        )
                        st.markdown("---")

                        gm1, gm2, gm3, gm4 = \
                            st.columns(4)
                        with gm1:
                            st.metric(
                                "Generated",
                                str(n_gen)
                            )
                        with gm2:
                            st.metric(
                                "Passing",
                                str(len(passing))
                            )
                        with gm3:
                            rate = len(passing) * \
                                100 // n_gen
                            st.metric(
                                "Pass Rate",
                                f"{rate}%"
                            )
                        with gm4:
                            if passing:
                                st.metric(
                                    "Best EMI",
                                    f"{passing[0]['emi_dbm']:.2f} dBm"
                                )

                        # Top designs table
                        if passing:
                            st.markdown(
                                "### 🏆 Top 10 "
                                "Generated Designs"
                            )
                            top_data = []
                            for i, d in enumerate(
                                passing[:10]
                            ):
                                top_data.append({
                                    'Rank':
                                        f"#{i+1}",
                                    'Width (mm)':
                                        f"{d['trace_width_mm']:.2f}",
                                    'Length (mm)':
                                        f"{d['trace_length_mm']:.1f}",
                                    'Vias':
                                        str(int(d[
                                            'stitching_vias'
                                        ])),
                                    'Decap (mm)':
                                        f"{d['decap_distance_mm']:.1f}",
                                    'EMI (dBm)':
                                        f"{d['emi_dbm']:.2f}",
                                    'Margin':
                                        f"{d['margin']:.2f} dBm"
                                })
                            st.dataframe(
                                pd.DataFrame(
                                    top_data
                                ),
                                hide_index=True,
                                width='stretch'
                            )

                            # EMI distribution chart
                            fig_gen, ax_gen = \
                                plt.subplots(
                                    figsize=(10, 4)
                                )
                            all_emis = [
                                d['emi_dbm']
                                for d in all_designs
                            ]
                            pass_emis = [
                                d['emi_dbm']
                                for d in passing
                            ]
                            ax_gen.hist(
                                all_emis, bins=40,
                                color='lightblue',
                                alpha=0.7,
                                label=f'All ({n_gen})',
                                edgecolor='blue'
                            )
                            ax_gen.hist(
                                pass_emis, bins=20,
                                color='green',
                                alpha=0.7,
                                label=f'Passing ({len(passing)})',
                                edgecolor='darkgreen'
                            )
                            ax_gen.axvline(
                                x=LIMIT,
                                color='red',
                                linestyle='--',
                                linewidth=2,
                                label=f'Limit ({LIMIT} dBm)'
                            )
                            ax_gen.set_xlabel(
                                'Predicted EMI (dBm)'
                            )
                            ax_gen.set_ylabel('Count')
                            ax_gen.set_title(
                                'Generated Design '
                                'Distribution',
                                fontweight='bold'
                            )
                            ax_gen.legend()
                            ax_gen.grid(
                                True, alpha=0.3
                            )
                            st.pyplot(fig_gen)
                            plt.close()

                            # Save and download
                            top_10 = passing[:min(
                                10, len(passing)
                            )]
                            save_data = {
                                'total_generated':
                                    n_gen,
                                'total_passing':
                                    len(passing),
                                'pass_rate':
                                    len(passing) /
                                    n_gen,
                                'top_designs': top_10
                            }
                            with open(
                                gen_path, 'w'
                            ) as f:
                                json.dump(
                                    save_data,
                                    f, indent=2
                                )

                            st.download_button(
                                "⬇️ Download Top 10 Designs",
                                data=json.dumps(
                                    save_data,
                                    indent=2
                                ),
                                file_name=(
                                    "generated_designs"
                                    ".json"
                                ),
                                mime=(
                                    "application/json"
                                )
                            )

                    else:
                        st.warning(
                            "VAE model not found. "
                            "Run python models/"
                            "generative_designer.py "
                            "first!"
                        )

                except Exception as e:
                    st.error(f"Error: {e}")

        elif os.path.exists(gen_path):
            st.info(
                "Showing previous results. "
                "Click Generate for new designs."
            )
            with open(gen_path) as f:
                prev_gen = json.load(f)

            pg1, pg2, pg3 = st.columns(3)
            with pg1:
                st.metric(
                    "Total Generated",
                    str(prev_gen.get(
                        'total_generated', 0
                    ))
                )
            with pg2:
                st.metric(
                    "Passing Designs",
                    str(prev_gen.get(
                        'total_passing', 0
                    ))
                )
            with pg3:
                rate = prev_gen.get(
                    'pass_rate', 0
                )
                st.metric(
                    "Pass Rate",
                    f"{rate*100:.1f}%"
                )

            top = prev_gen.get('top_designs', [])
            if top:
                st.markdown(
                    "### 🏆 Top Generated Designs"
                )
                top_data = []
                for i, d in enumerate(top[:5]):
                    top_data.append({
                        'Rank':
                            f"#{i+1}",
                        'Width (mm)':
                            f"{d.get('trace_width_mm', 0):.2f}",
                        'Length (mm)':
                            f"{d.get('trace_length_mm', 0):.1f}",
                        'Vias':
                            str(int(d.get(
                                'stitching_vias', 0
                            ))),
                        'EMI (dBm)':
                            f"{d.get('emi_dbm', 0):.2f}",
                        'Margin':
                            f"{d.get('margin', 0):.2f} dBm"
                    })
                st.dataframe(
                    pd.DataFrame(top_data),
                    hide_index=True,
                    width='stretch'
                )

            if os.path.exists(gen_img):
                st.image(
                    gen_img,
                    caption=(
                        "Generative Designer Results"
                    ),
                    use_container_width=True
                )

        else:
            st.warning(
                "⚠️ Run this first:\n"
                "```\n"
                "python models/"
                "generative_designer.py"
                "\n```"
            )
# ─────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────

st.markdown("---")
st.markdown(
    """
    <div style='text-align:center;
                color:#888;
                font-size:0.85rem;'>
    🛡️ NeuroShield-PCB &nbsp;|&nbsp;
    KAN-PINN Surrogate &nbsp;|&nbsp;
    5 Real Physics Models &nbsp;|&nbsp;
    Multi-Agent DRL &nbsp;|&nbsp;
    Federated Learning &nbsp;|&nbsp;
    Digital Twin &nbsp;|&nbsp;
    KiCad Integration &nbsp;|&nbsp;
    Uncertainty AI &nbsp;|&nbsp;
    Fourier Operator &nbsp;|&nbsp;
    Generative Design &nbsp;|&nbsp;
    Built for EMI/EMC  Project
    </div>
    """,
    unsafe_allow_html=True
)
