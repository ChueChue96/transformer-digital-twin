# 3_digital_twin.py
"""
Digital Twin Core Class
Purpose: Maintain twin state, compute Health Index, detect anomalies
"""
from datetime import datetime
import numpy as np
import pandas as pd
import os

os.makedirs("data", exist_ok=True)


class TransformerDigitalTwin:
    """
    Digital Twin for a single transformer.

    Receives predicted (physics) and measured (sensor) temperatures,
    calculates residual, detects anomalies, and computes Health Index.
    """

    def __init__(self, name, rated_power_kva=1000):
        self.name = name
        self.rated_power_kva = rated_power_kva
        self.history = []
        self.residuals = []
        self.hi_history = []

        # IEC 60076-7 alarm thresholds
        self.oil_temp_alarm = 90       # °C
        self.hotspot_temp_alarm = 105  # °C

        print(f"✅ Twin initialized: {name} ({rated_power_kva} kVA)")

    def update(self, timestamp, predicted_temp, measured_temp,
               load_percent, ambient_temp):
        """Update twin with one new reading."""
        residual = measured_temp - predicted_temp
        self.residuals.append(residual)

        # Health Index: 100 − 5 × |deviation from mean residual|
        if len(self.residuals) >= 5:
            mean_r = np.mean(self.residuals[:-1])
            deviation = abs(residual - mean_r)
            hi = max(0.0, 100.0 - deviation * 5.0)
        else:
            hi = 100.0
        self.hi_history.append(hi)

        record = {
            "timestamp": timestamp,
            "predicted": predicted_temp,
            "measured": measured_temp,
            "residual": residual,
            "load_%": load_percent,
            "ambient_°C": ambient_temp,
            "health_index": hi,
        }
        self.history.append(record)
        return record

    def check_anomaly(self):
        """3-sigma anomaly detection on residual."""
        if len(self.residuals) < 10:
            return []

        recent = self.residuals[-10:]
        mean_r = np.mean(recent)
        std_r = np.std(recent)
        current = self.residuals[-1]

        alerts = []
        if std_r > 0.1 and abs(current - mean_r) > 3 * std_r:
            alerts.append(
                f"⚠️ RESIDUAL ANOMALY: {current:+.2f} °C "
                f"(expected {mean_r:+.2f} ± {3*std_r:.2f})"
            )
        return alerts

    def check_thermal(self, predicted_temp):
        """Check against absolute thermal limits."""
        alerts = []
        if predicted_temp > self.hotspot_temp_alarm:
            alerts.append(f"🔥 HOT-SPOT HIGH: {predicted_temp:.1f} °C")
        elif predicted_temp > self.oil_temp_alarm:
            alerts.append(f"⚠️ OIL TEMP HIGH: {predicted_temp:.1f} °C")
        return alerts

    def summary(self):
        if not self.history:
            return {}
        df = pd.DataFrame(self.history)
        return {
            "name": self.name,
            "samples": len(df),
            "mean_residual": round(df["residual"].mean(), 3),
            "std_residual": round(df["residual"].std(), 3),
            "max_residual": round(df["residual"].max(), 3),
            "min_health_index": round(df["health_index"].min(), 1),
            "current_health_index": round(df["health_index"].iloc[-1], 1),
        }


# ============ Main: Run through residual data ============
if __name__ == "__main__":
    # Load data from Step 2
    df = pd.read_csv("data/residuals.csv", index_col=0, parse_dates=True)
    print(f"✅ Loaded residual data: {len(df)} rows")

    twin = TransformerDigitalTwin("Enertech-TX-001", rated_power_kva=1000)

    print(f"\n=== Processing {len(df)} timesteps ===")

    for i, (ts, row) in enumerate(df.iterrows()):
        # Approximate load and ambient from the profile (placeholders)
        load_percent = 70.0
        ambient_temp = 28.0

        record = twin.update(
            timestamp=ts,
            predicted_temp=row["predicted"],
            measured_temp=row["measured"],
            load_percent=load_percent,
            ambient_temp=ambient_temp,
        )

        # Check both anomaly types
        anomaly_alerts = twin.check_anomaly()
        thermal_alerts = twin.check_thermal(row["predicted"])

        for alert in anomaly_alerts + thermal_alerts:
            print(f"[{ts}] {alert}")

        # Progress every 24 steps (6 hours)
        if (i + 1) % 24 == 0:
            print(f"  Processed {i+1}/{len(df)} | HI = {record['health_index']:.1f}")

    # Save twin history
    history_df = pd.DataFrame(twin.history)
    history_df.to_csv("data/twin_history.csv", index=False)

    print(f"\n=== Twin Summary ===")
    for k, v in twin.summary().items():
        print(f"   {k}: {v}")
    print(f"\n💾 Saved to data/twin_history.csv")