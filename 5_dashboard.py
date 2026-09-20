# 5_dashboard.py
"""
Streamlit Dashboard - Demo UI
Run: streamlit run 5_dashboard.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(
    page_title="Transformer Digital Twin",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Transformer Digital Twin — Enertech Demo")
st.caption("Physics-based hot-spot prediction with residual-based anomaly detection")

# ============ Load Data ============
try:
    predicted = pd.read_csv("data/predicted_temps.csv",
                            index_col=0, parse_dates=True)
    residuals = pd.read_csv("data/residuals.csv",
                            index_col=0, parse_dates=True)
    twin_history = pd.read_csv("data/twin_history.csv",
                               parse_dates=["timestamp"])
except FileNotFoundError:
    st.error("❌ Data not found. Run steps 1, 2, 3 first.")
    st.stop()

# ============ KPI Cards ============
col1, col2, col3, col4 = st.columns(4)

with col1:
    peak_oil = predicted["top_oil_temp"].max()
    st.metric("Peak Top-oil", f"{peak_oil:.1f} °C",
              delta=f"{peak_oil - 90:.1f} to alarm")

with col2:
    peak_hs = predicted["hot_spot_temp"].max()
    st.metric("Peak Hot-spot", f"{peak_hs:.1f} °C",
              delta=f"{peak_hs - 105:.1f} to alarm")

with col3:
    std_r = residuals["residual"].std()
    st.metric("Residual Std", f"{std_r:.2f} °C",
              help="Lower is better. <2 °C = good")

with col4:
    current_hi = twin_history["health_index"].iloc[-1]
    min_hi = twin_history["health_index"].min()
    st.metric("Health Index (min)", f"{min_hi:.1f}",
              delta=f"current {current_hi:.1f}")

# ============ Figure 1: Temperature + Residual ============
st.subheader("📈 Thermal Behavior")

fig = make_subplots(
    rows=2, cols=1, shared_xaxes=True,
    subplot_titles=("Temperature Prediction", "Residual"),
    vertical_spacing=0.12, row_heights=[0.6, 0.4]
)

fig.add_trace(
    go.Scatter(x=predicted.index, y=predicted["top_oil_temp"],
               name="Top-oil (predicted)",
               line=dict(color="blue", width=2)),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=predicted.index, y=predicted["hot_spot_temp"],
               name="Hot-spot (predicted)",
               line=dict(color="red", width=2)),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=residuals.index, y=residuals["measured"],
               name="Measured oil",
               line=dict(color="green", dash="dash")),
    row=1, col=1
)
fig.add_hline(y=90, line_dash="dot", line_color="orange", row=1, col=1)
fig.add_hline(y=105, line_dash="dot", line_color="darkred", row=1, col=1)

fig.add_trace(
    go.Scatter(x=residuals.index, y=residuals["residual"],
               name="Residual",
               line=dict(color="purple", width=1.5)),
    row=2, col=1
)
fig.add_hline(y=0, line_dash="dash", line_color="black", row=2, col=1)

fig.update_yaxes(title_text="Temperature (°C)", row=1, col=1)
fig.update_yaxes(title_text="Residual (°C)", row=2, col=1)
fig.update_layout(height=600, hovermode="x unified")

st.plotly_chart(fig, use_container_width=True)

# ============ Figure 2: Health Index ============
st.subheader("💚 Health Index")

fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=twin_history["timestamp"],
    y=twin_history["health_index"],
    fill="tozeroy",
    line=dict(color="darkgreen", width=2),
    name="Health Index"
))
fig2.add_hline(y=90, line_dash="dot", line_color="orange",
               annotation_text="Warning")
fig2.add_hline(y=70, line_dash="dot", line_color="red",
               annotation_text="Critical")
fig2.update_layout(
    yaxis=dict(range=[0, 105], title="Health Index"),
    height=300, hovermode="x unified"
)
st.plotly_chart(fig2, use_container_width=True)

# ============ Anomaly Table ============
st.subheader("🚨 Detected Anomalies")

std_r = residuals["residual"].std()
mean_r = residuals["residual"].mean()
anomalies = residuals[abs(residuals["residual"] - mean_r) > 3 * std_r]

if len(anomalies) > 0:
    anomaly_df = anomalies.copy()
    anomaly_df["severity"] = (abs(anomaly_df["residual"] - mean_r) / std_r).round(2)
    st.dataframe(anomaly_df, use_container_width=True)
else:
    st.success("✅ No anomalies detected")

# ============ About ============
with st.expander("ℹ️ About this Digital Twin"):
    st.markdown("""
    **Architecture:**
    - **Layer 1 (Physics)**: IEC 60076-7 thermal model
    - **Layer 2 (Residual)**: residual = measured − predicted
    - **Layer 3 (AI)**: 3-sigma anomaly detection

    **Novelty**: The physics residual feeds anomaly detection,
    not the temperature itself.

    **Limitations** (per Appendix 1):
    - Hot-spot accuracy cannot be validated without fibre-optic sensors
    - RUL in months/years requires labelled failure events (none exist)
    - Batch twin only — not real-time
    """)