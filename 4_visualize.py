# 4_visualize.py
"""
Visualization Layer
Purpose: Create figures for report and Enertech demo
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs("figures", exist_ok=True)

# ============ Load Data ============
predicted = pd.read_csv("data/predicted_temps.csv",
                        index_col=0, parse_dates=True)
residuals = pd.read_csv("data/residuals.csv",
                        index_col=0, parse_dates=True)
twin_history = pd.read_csv("data/twin_history.csv",
                           parse_dates=["timestamp"])

print(f"✅ Loaded {len(predicted)} predicted rows")
print(f"✅ Loaded {len(residuals)} residual rows")
print(f"✅ Loaded {len(twin_history)} twin history rows")

# ============ Figure 1: Temperature Time Series ============
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(predicted.index, predicted["top_oil_temp"],
        label="Top-oil (predicted)", color="blue", linewidth=2)
ax.plot(predicted.index, predicted["hot_spot_temp"],
        label="Hot-spot (predicted)", color="red", linewidth=2)
ax.plot(residuals.index, residuals["measured"],
        label="Oil (measured)", color="green",
        linestyle="--", linewidth=1.5)
ax.axhline(y=90, color="orange", linestyle=":",
           label="Oil alarm (90 °C)")
ax.axhline(y=105, color="darkred", linestyle=":",
           label="Hot-spot alarm (105 °C)")
ax.set_xlabel("Time")
ax.set_ylabel("Temperature (°C)")
ax.set_title("Transformer Thermal Behavior — IEC 60076-7 Model")
ax.legend(loc="upper left")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/fig1_temperature.png", dpi=150)
print("✅ Saved figures/fig1_temperature.png")

# ============ Figure 2: Residual Analysis ============
fig, axes = plt.subplots(2, 1, figsize=(14, 7))

mean_r = residuals["residual"].mean()
std_r = residuals["residual"].std()

# Top: residual time series
axes[0].plot(residuals.index, residuals["residual"],
             color="purple", linewidth=1.5)
axes[0].axhline(0, color="black", linestyle="--", alpha=0.5)
axes[0].axhline(mean_r + 3*std_r, color="red", linestyle=":",
                label=f"±3σ ({3*std_r:.2f} °C)")
axes[0].axhline(mean_r - 3*std_r, color="red", linestyle=":")
axes[0].set_ylabel("Residual (°C)")
axes[0].set_title("Residual = Measured − Predicted")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Bottom: histogram
axes[1].hist(residuals["residual"], bins=30,
             color="purple", alpha=0.7)
axes[1].axvline(mean_r, color="black", linestyle="--",
                label=f"Mean = {mean_r:.2f}")
axes[1].axvline(mean_r + 3*std_r, color="red",
                linestyle=":", label="±3σ")
axes[1].axvline(mean_r - 3*std_r, color="red", linestyle=":")
axes[1].set_xlabel("Residual (°C)")
axes[1].set_ylabel("Frequency")
axes[1].set_title("Residual Distribution")
axes[1].legend()

plt.tight_layout()
plt.savefig("figures/fig2_residual.png", dpi=150)
print("✅ Saved figures/fig2_residual.png")

# ============ Figure 3: Health Index ============
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(twin_history["timestamp"], twin_history["health_index"],
        color="darkgreen", linewidth=2)
ax.fill_between(twin_history["timestamp"],
                twin_history["health_index"], 100,
                alpha=0.2, color="green")
ax.axhline(y=90, color="orange", linestyle=":",
           label="Warning (90)")
ax.axhline(y=70, color="red", linestyle=":",
           label="Critical (70)")
ax.set_xlabel("Time")
ax.set_ylabel("Health Index")
ax.set_title("Transformer Health Index (from residual)")
ax.set_ylim(0, 105)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("figures/fig3_health_index.png", dpi=150)
print("✅ Saved figures/fig3_health_index.png")

print("\n✅ All figures created. Check figures/ folder.")