import sys

import matplotlib.pyplot as plt
import pandas as pd

# Pass a different CSV as the first argument if needed: python plot.py run1.csv
csv_filename = sys.argv[1] if len(sys.argv) > 1 else "capture.csv"

try:
    df = pd.read_csv(csv_filename)
except FileNotFoundError:
    sys.exit(
        f"Error: Could not find '{csv_filename}'. Run the capture script first, "
        "from this same folder, or pass the CSV path as an argument."
    )

# Firmware-computed totals (whole watts); everything else ending in _W is a rail.
AGGREGATES = ["EPS_W", "PCIE_W", "MB_W", "TOTAL_W"]
rail_cols = [c for c in df.columns if c.endswith("_W") and c not in AGGREGATES]

# Hide rails that never drew meaningful power (e.g. unused PCIE3 or HPWR wires).
MIN_ACTIVE_W = 0.05
active = [c for c in rail_cols if df[c].abs().max() >= MIN_ACTIVE_W]
idle = [c[:-2] for c in rail_cols if c not in active]
print("Plotting rails:", [c[:-2] for c in active])
if idle:
    print("Skipped idle rails:", idle)

t = df["Time_s"]
duration = t.iloc[-1] - t.iloc[0]
rate = (len(df) - 1) / duration if duration > 0 else float("nan")

fig, (ax_rails, ax_total) = plt.subplots(
    2, 1, figsize=(12, 8), sharex=True,
    gridspec_kw={"height_ratios": [2, 1]}, constrained_layout=True,
)

for col in active:
    ax_rails.plot(t, df[col], label=col[:-2], alpha=0.8, linewidth=1)
ax_rails.set_ylabel("Rail power (W)", fontsize=12)
ax_rails.set_title(
    f"PMD2X High-Speed Power Monitoring ({len(df)} samples, ~{rate:.0f} Hz)", fontsize=14
)
ax_rails.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
ax_rails.grid(True, linestyle="--", alpha=0.6)

for col in AGGREGATES:
    if col in df.columns:
        ax_total.plot(t, df[col], label=col[:-2], alpha=0.8, linewidth=1)
ax_total.set_xlabel("Time (s)", fontsize=12)
ax_total.set_ylabel("Aggregate power (W)", fontsize=12)
ax_total.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
ax_total.grid(True, linestyle="--", alpha=0.6)

print("Displaying graph...")
plt.show()