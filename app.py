import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from data_prep import generate_mock_ecg, delta_modulation_encoder
from model import get_pretrained_mock, run_inference

st.set_page_config(page_title="NeuroFit Dashboard", layout="wide")

st.title(" NeuroFit: Ultra-Low-Power Edge ECG Neuromorphic Diagnostics")
st.markdown("---")


if "nn_model" not in st.session_state:
    st.session_state.nn_model = get_pretrained_mock()


st.sidebar.header(" Edge Patient Profiles")
patient_selection = st.sidebar.selectbox(
    "Select Target Patient Stream:",
    ["Patient 102 (Normal Basal Rhythm)", "Patient 209 (Severe Atrial Arrhythmia)"]
)

threshold_val = st.sidebar.slider("Delta Modulation Sensitivity Threshold (𝚫V)", 0.05, 0.40, 0.15)


profile_mode = "normal" if "Normal" in patient_selection else "arrhythmia"
raw_ecg = generate_mock_ecg(profile_mode)
spike_train = delta_modulation_encoder(raw_ecg, threshold=threshold_val)


prediction_idx, metrics_sparsity = run_inference(st.session_state.nn_model, spike_train)
synops_reduction = metrics_sparsity * 100.0  


col1, col2, col3 = st.columns(3)
with col1:
    st.metric("System Diagnostic Status", "Active Monitoring")
with col2:
    st.metric("Sparsity Matrix Metric", f"{metrics_sparsity*100:.1f} %")
with col3:
    st.metric("Theoretical Energy Savings", f"~ {synops_reduction:.1f} % Less Power")


if prediction_idx == 1:
    st.error("🚨 CRITICAL ALERT: Irregular Ventricular Spacing Caught. Anomaly Firing High-Frequency Output.")
else:
    st.success("✅ SYSTEM STABLE: Consistent Basal Rhythmic Trace Found. Standby Energy Save Loop On.")

st.subheader("📈 Live Telemetry Signal Stream Mapping")


fig = go.Figure()
fig.add_trace(go.Scatter(y=raw_ecg, mode='lines', name='Continuous ECG Voltage (mV)', line=dict(color='#00CC96', width=2)))
fig.update_layout(margin=dict(l=20, r=20, t=20, b=20), height=300)
st.plotly_chart(fig, use_container_width=True)


st.subheader("🧬 Neuromorphic Event-Driven Spike Train Matrices")
spike_display = spike_train.squeeze(1).numpy().T 

fig_spk = go.Figure(data=go.Heatmap(
    z=spike_display,
    x=list(range(len(raw_ecg))),
    y=['Positive Changes', 'Negative Drops'],
    colorscale='Hot',
    showscale=False
))
fig_spk.update_layout(height=180, margin=dict(l=20, r=20, t=10, b=10))
st.plotly_chart(fig_spk, use_container_width=True)