# 2_residual_calc.py
"""
Residual Calculation Layer
Purpose: Compute the difference between measured and predicted temperature
This residual is the input to anomaly detection
"""
import pandas as pd
import numpy as np
import os

os.makedirs("data", exist_ok=True)

# ============ Step 1: Load Predicted Data ============
predicted = pd.read_csv(
    "data/predicted_temps.csv",
    index_col=0,
    parse_dates=True
)
print(f"✅ Loaded predictions: {len(predicted)} rows")
print(f"   Columns: {list(predicted.columns)}")

# ============ Step 2: Load or Simulate Measured Data ============
# In the real project, this comes from Enertech as a CSV file:
#   timestamp, oil_temp
#
# For now, we simulate sensor noise around the predicted top-oil,
# and inject one anomaly so we can verify detection works.

np.random.seed(42)

# Base measurement = predicted top-oil + sensor noise
measured_oil_temp = (
    predicted["top_oil_temp"].values
    + np.random.normal(0, 0.8, len(predicted))
)

# Inject an anomaly at hour 14 (index 56 of 96 timesteps)
anomaly_idx = 56
measured_oil_temp[anomaly_idx] += 8.0
print(f"⚠️ Injected anomaly at index {anomaly_idx} (+8.0 °C)")

measured = pd.DataFrame(
    {"oil_temp": measured_oil_temp},
    index=predicted.index
)
measured.to_csv("data/measured_oil_temp.csv")
print(f"✅ Measured data created (simulated for now)")

# ============ Step 3: Calculate Residual ============
# THE KEY LINE: residual = measured − predicted
residual = measured["oil_temp"] - predicted["top_oil_temp"]

residual_mean = np.mean(residual)
residual_std = np.std(residual)

print(f"\n=== Residual Statistics ===")
print(f"   Mean: {residual_mean:+.3f} °C")
print(f"   Std:  {residual_std:.3f} °C")
print(f"   Min:  {residual.min():+.3f} °C")
print(f"   Max:  {residual.max():+.3f} °C")

# ============ Step 4: Anomaly Detection (3-sigma) ============
threshold = 3 * residual_std
anomalies = residual[abs(residual - residual_mean) > threshold]

print(f"\n=== Anomaly Detection (3σ = {threshold:.2f} °C) ===")
if len(anomalies) > 0:
    print(f"⚠️ {len(anomalies)} anomalies detected:")
    for idx, val in anomalies.items():
        print(f"   {idx}: residual = {val:+.2f} °C")
else:
    print("✅ No anomalies detected")

# ============ Step 5: Save Combined Results ============
residual_df = pd.DataFrame({
    "predicted": predicted["top_oil_temp"],
    "measured": measured["oil_temp"],
    "residual": residual
})
residual_df.to_csv("data/residuals.csv")
print(f"\n💾 Saved to data/residuals.csv")