"""
Make the static charts used in README.md (PNG files in ./assets).

Reads the CSVs produced by phase1/phase2 and saves 4 plain-English charts.
Run from the project folder:   python make_charts.py
Needs:  pip install matplotlib
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import phase2_model as pm

PROC = Path("data/processed")
OUT = Path("assets")
OUT.mkdir(exist_ok=True)

NAVY, ORANGE, GRAY, TEAL = "#1f3a5f", "#e07b39", "#8a8f98", "#2a9d8f"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 150})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    print("saved", OUT / name)


hist = pd.read_csv(PROC / "history_totals.csv")
ft = pd.read_csv(PROC / "forecast_totals.csv")
fg = pd.read_csv(PROC / "forecast_by_grade.csv")
fg["grade"] = fg["grade"].astype(str)
bt = pd.read_csv(PROC / "backtest.csv")
m, ungr = pm.load_matrix()
last_year = int(hist.year.iloc[-1])
last_total = float(hist.total.iloc[-1])

# 1) History + forecast ------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5.2))
ax.plot(hist.year, hist.total, color=NAVY, lw=2.5, marker="o", label="Actual enrollment")
ax.fill_between(ft.year, ft.p10, ft.p90, color=ORANGE, alpha=0.25, label="80% range of outcomes")
xs = [last_year] + ft.year.tolist()
ax.plot(xs, [last_total] + ft["median"].tolist(), color=ORANGE, lw=2.5, marker="o", label="Model's middle estimate")
ax.plot(xs, [last_total] * len(xs), color=GRAY, lw=2, ls="--", label="'Nothing changes' baseline")
ax.set_title("LCUSD enrollment: 12 years of history and a 5-year forecast", loc="left", fontweight="bold")
ax.set_ylabel("Students (K-12)")
ax.set_xlabel("School year (ending)")
ax.legend(frameon=False, loc="best")
save(fig, "1_history_and_forecast.png")

# 2) Do grades grow or shrink as they move up? ---------------------------------------
csr = pm.csr_table(m)
csr = csr.drop(index=[y for y in csr.index if y in pm.EXCLUDE]).mean()
labels = [f"{a}→{b}" for a, b in zip(pm.GRADES[:-1], pm.GRADES[1:])]
fig, ax = plt.subplots(figsize=(10, 5.2))
pct = (csr.values - 1) * 100
ax.bar(labels, pct, color=[TEAL if p >= 0 else ORANGE for p in pct])
ax.axhline(0, color="black", lw=1)
ax.set_title("What happens to a class when it moves up a grade?", loc="left", fontweight="bold")
ax.set_ylabel("Typical change in class size (%)")
ax.set_xlabel("Grade transition (e.g. 3→4 = this year's 3rd graders next year)")
ax.text(0.99, 0.97, "Green = class gains students\nOrange = class loses students",
        transform=ax.transAxes, ha="right", va="top", fontsize=10, color=GRAY)
save(fig, "2_class_size_change_by_grade.png")

# 3) Today vs forecast by grade ------------------------------------------------------
final = fg[fg.year == fg.year.max()].set_index("grade").loc[pm.GRADES]
now = m.iloc[-1]
x = np.arange(len(pm.GRADES))
fig, ax = plt.subplots(figsize=(10, 5.2))
ax.bar(x - 0.2, now.values, 0.4, color=NAVY, label=f"{last_year} (actual)")
ax.bar(x + 0.2, final["median"].values, 0.4, color=ORANGE, label=f"{int(fg.year.max())} (forecast)")
ax.set_xticks(x)
ax.set_xticklabels(pm.GRADES)
ax.set_xlabel("Grade")
ax.set_ylabel("Students")
ax.set_title("Students per grade: today vs. forecast", loc="left", fontweight="bold")
ax.legend(frameon=False)
save(fig, "3_by_grade_today_vs_forecast.png")

# 4) How accurate has it been? -------------------------------------------------------
g = bt.groupby("horizon")[["model_ape", "naive_ape"]].mean()
g.loc["All"] = bt[["model_ape", "naive_ape"]].mean()
names = [f"{h} yr ahead" if h != "All" else "All forecasts" for h in g.index]
x = np.arange(len(g))
fig, ax = plt.subplots(figsize=(10, 5.2))
b1 = ax.bar(x - 0.2, g.naive_ape, 0.4, color=GRAY, label="Baseline: 'nothing changes'")
b2 = ax.bar(x + 0.2, g.model_ape, 0.4, color=ORANGE, label="Cohort model")
ax.bar_label(b1, fmt="%.1f%%", fontsize=10)
ax.bar_label(b2, fmt="%.1f%%", fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylabel("Average error (%) — lower is better")
ax.set_title(f"Backtest: how wrong were past forecasts? ({len(bt)} forecasts)", loc="left", fontweight="bold")
ax.legend(frameon=False)
save(fig, "4_backtest_accuracy.png")

# Numbers to paste into the README -----------------------------------------------------
f = ft.iloc[-1]
print("\n--- README numbers ---")
print(f"{last_year} students: {last_total:,.0f}")
print(f"{int(f.year)} median: {f['median']:,.0f}   80% range: {f.p10:,.0f}-{f.p90:,.0f}   "
      f"change vs {last_year}: {(f['median'] / last_total - 1) * 100:+.1f}%")
print(f"Backtest MAPE  model {bt.model_ape.mean():.2f}%  vs baseline {bt.naive_ape.mean():.2f}%  (n={len(bt)})")