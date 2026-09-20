# 1_physics_model.py
"""
Physics Model Layer - IEC 60076-7 thermal model
Purpose: Predict transformer hot-spot and top-oil temperature
"""
import pandas as pd
from transformer_thermal_model.model import Model
from transformer_thermal_model.cooler import CoolerType
from transformer_thermal_model.schemas import (
    UserTransformerSpecifications, InputProfile
)
from transformer_thermal_model.transformer import PowerTransformer
import os

# Folder ဖန်တီး
os.makedirs("data", exist_ok=True)

# ============ Step 1: Transformer Specifications ============
# ဒီတန်ဖိုးတွေက Enertech ရဲ့ Factory Test Report ကနေ ရမယ်
# လောလောဆယ် placeholder တန်ဖိုး သုံးထားတယ်
specs = UserTransformerSpecifications(
    load_loss=1000,           # W - copper loss at rated load
    nom_load_sec_side=1500,   # A - rated secondary current
    no_load_loss=200,         # W - core loss
    amb_temp_surcharge=20     # K - ambient temp surcharge
)

transformer = PowerTransformer(
    user_specs=specs,
    cooling_type=CoolerType.ONAN  # Oil Natural Air Natural
)

print(f"✅ Transformer created")
print(f"   Load loss: {specs.load_loss} W")
print(f"   No-load loss: {specs.no_load_loss} W")
print(f"   Cooling: ONAN")

# ============ Step 2: Load & Ambient Profile ============
# 24 နာရီ၊ 15 မိနစ် resolution
idx = pd.date_range("2026-09-21", periods=96, freq="15min")

# Commercial load profile (မနက် ၈ နာရီကနေ ညနေ ၆ နာရီ အထိ peak)
load_profile = []
ambient_profile = []
for i in range(96):
    hour = i / 4  # 0-24
    # Load: peak during working hours
    if 8 <= hour <= 18:
        load = 800 + 200 * ((hour - 13) / 5) ** 2
    else:
        load = 300 + 100 * (hour / 24)
    load_profile.append(load)
    
    # Ambient: peak at 14:00
    ambient = 25 + 8 * (1 - abs(hour - 14) / 12)
    ambient_profile.append(ambient)

profile = InputProfile.create(
    datetime_index=idx,
    load_profile=pd.Series(load_profile, index=idx),
    ambient_temperature_profile=pd.Series(ambient_profile, index=idx)
)

print(f"✅ Profile created: {len(profile.datetime_index)} timesteps")

# ============ Step 3: Run Physics Model ============
model = Model(temperature_profile=profile, transformer=transformer)
result = model.run()

print(f"\n✅ Physics model executed successfully")

# Access via attributes (NOT brackets)
top_oil = result.top_oil_temp_profile
hot_spot = result.hot_spot_temp_profile

print(f"\n=== Predicted Temperatures (first 5 rows) ===")
print(top_oil.head())
print(hot_spot.head())

print(f"\n=== Peak values ===")
print(f"   Max top-oil temp:  {top_oil.max():.2f} °C")
print(f"   Max hot-spot temp: {hot_spot.max():.2f} °C")

# Save as DataFrame
df = pd.DataFrame({
    "top_oil_temp": top_oil,
    "hot_spot_temp": hot_spot
})
df.to_csv("data/predicted_temps.csv")
print(f"\n💾 Saved to data/predicted_temps.csv")