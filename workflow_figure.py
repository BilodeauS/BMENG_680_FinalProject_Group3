"""Workflow figure: original data -> chronological split -> models -> tests on held-out data.
All forecast numbers are recomputed from the saved result files; hot-spot and hourly numbers come from
validate_hotspots.py / time_of_day.py output (hard-coded below, with the source noted)."""
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import font_manager as fm

have = {f.name for f in fm.fontManager.ttflist}
plt.rcParams["font.family"] = next(f for f in ["Times New Roman", "Liberation Serif", "DejaVu Serif"] if f in have)

# ---------- numbers ----------
p = pd.read_csv("Forecasting/daily_test_predictions.csv", index_col=0, parse_dates=True)
mean = p.actual.mean()
cols = {"b1": "Baseline: day-of-week average", "b2": "Baseline: day-of-week x month average",
        "poi": "Poisson regression", "rf": "Random forest"}
mae = {k: (p.actual - p[c]).abs().mean() for k, c in cols.items()}
tb = pd.read_csv("Validations/time_block_forecast_summary.csv").set_index("model")["MAE (incidents per block)"]
blk_base, blk_poi = tb["Baseline: block x day-of-week avg"], tb["Poisson regression"]
n_test = len(p)
# from validate_hotspots.py output
HS_TRAIN, HS_TEST, HS_RANDOM, TOP10_REAPPEAR = 12.0, 12.6, 1.4, 9
# from time_of_day.py output
R_WEEKDAY, R_WEEKEND = 0.994, 0.970

# ---------- style ----------
GREY_F, GREY_E = "#EDEDED", "#777777"
BLUE_F, BLUE_E = "#DCE8F6", "#2E6DB0"
AMB_F, AMB_E, AMB_T = "#FFF5DC", "#B7791F", "#8A5A10"
INK, MUTED = "#1A1A1A", "#555555"

W, H = 7.8, 6.6
fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

def box(x, y, w, h, fc, ec, lw=1.0, ls="-", r=0.06, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))
def arrow(a, b, color=MUTED, lw=1.2, ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, color=color, lw=lw, ls=ls, zorder=4, shrinkA=0, shrinkB=0))
def line(xs, ys, color=MUTED, lw=1.2, ls="-"):
    ax.plot(xs, ys, color=color, lw=lw, ls=ls, zorder=3, solid_capstyle="butt")
def txt(x, y, s, size=8, weight="normal", ha="center", color=INK, style="normal"):
    ax.text(x, y, s, fontsize=size, fontweight=weight, ha=ha, va="center", color=color, fontstyle=style, zorder=5)

# ---------- geometry ----------
LX, LW = 0.10, 1.55            # data column
MX, MW = 2.10, 2.20            # model column
EX, EW = 4.75, 2.75            # evaluation column
BUS_TRAIN, BUS_TEST = 1.88, 7.70
R1 = (3.45, 6.05); R2 = (2.15, 3.30); R3 = (0.85, 2.00)
mid = lambda r: (r[0] + r[1]) / 2

# column headers
txt(LX + LW / 2, 6.36, "Data", 9, "bold", color=MUTED)
txt(MX + MW / 2, 6.36, "Models (fit on 2022–2024)", 9, "bold", color=MUTED)
txt(EX + EW / 2, 6.36, "Tests on held-out 2025–2026 data", 9, "bold", color=MUTED)

# ---------- data column ----------
box(LX, 5.00, LW, 1.05, GREY_F, GREY_E)
txt(LX + LW / 2, 5.80, "Calgary Traffic", 8, "bold"); txt(LX + LW / 2, 5.64, "Incidents dataset", 8, "bold")
txt(LX + LW / 2, 5.36, "Open Calgary", 8); txt(LX + LW / 2, 5.20, "66,073 records, 2016–2026", 7.5)
arrow((LX + LW / 2, 5.00), (LX + LW / 2, 4.67))

box(LX, 3.55, LW, 1.10, GREY_F, GREY_E)
txt(LX + LW / 2, 4.47, "Cleaning", 8, "bold")
for i, s in enumerate(["quadrants filled", "UTC → Calgary time", "recording gaps removed", "2022 onward kept"]):
    txt(LX + LW / 2, 4.27 - i * 0.17, s, 7.5)
arrow((LX + LW / 2, 3.55), (LX + LW / 2, 3.32))

box(LX, 0.45, LW, 2.85, "none", "#AAAAAA", lw=0.9, ls=(0, (3, 2)), z=1)
txt(LX + LW / 2, 3.16, "Split by date", 7.5, color=MUTED, style="italic")
box(LX + 0.08, 1.88, LW - 0.16, 1.10, BLUE_F, BLUE_E)
txt(LX + LW / 2, 2.80, "Training set", 8, "bold")
for i, s in enumerate(["2022–2024", "1,096 days", "22,293 incidents"]):
    txt(LX + LW / 2, 2.60 - i * 0.17, s, 7.5)
box(LX + 0.08, 0.57, LW - 0.16, 1.10, AMB_F, AMB_E)
txt(LX + LW / 2, 1.49, "Test set (held out)", 8, "bold", color=AMB_T)
for i, s in enumerate(["2025–2026", "645 days", "14,228 incidents"]):
    txt(LX + LW / 2, 1.29 - i * 0.17, s, 7.5)
txt(LX + LW / 2, 0.69, "May–Jul 2025 gap excluded", 6.8, color=MUTED, style="italic")
txt(LX + LW / 2, 2.00, "used to fit every model", 6.8, color=MUTED, style="italic")

# training data -> models (solid bus)
ty = 2.43
line([LX + LW - 0.08, BUS_TRAIN], [ty, ty])
line([BUS_TRAIN, BUS_TRAIN], [mid(R3), mid(R1)])
for r in (R1, R2, R3):
    arrow((BUS_TRAIN, mid(r)), (MX, mid(r)))

# ---------- model column ----------
def group(r, title, sub):
    box(MX, r[0], MW, r[1] - r[0], "white", "#BBBBBB")
    txt(MX + MW / 2, r[1] - 0.20, title, 8.5, "bold"); txt(MX + MW / 2, r[1] - 0.38, sub, 7.5, color=MUTED, style="italic")
group(R1, "Daily incident counts", "How many incidents per day?")
group(R2, "Hot spots", "Where do incidents cluster?")
group(R3, "Time of day", "When are incidents busiest?")

def small(x, y, w, h, fc, ec, lines):
    box(x, y, w, h, fc, ec)
    n = len(lines)
    for i, (s, wt) in enumerate(lines):
        txt(x + w / 2, y + h / 2 + (n - 1) * 0.085 - i * 0.17, s, 8 if wt == "bold" else 7.5, wt)
sw, sh = 0.98, 0.88
small(MX + 0.08, 4.50, sw, sh, GREY_F, GREY_E, [("Baseline 1", "bold"), ("day-of-week", "normal"), ("average", "normal")])
small(MX + 0.08 + sw + 0.08, 4.50, sw, sh, GREY_F, GREY_E, [("Baseline 2", "bold"), ("day-of-week ×", "normal"), ("month average", "normal")])
small(MX + 0.08, 3.52, sw, sh, BLUE_F, BLUE_E, [("Poisson", "bold"), ("regression", "normal")])
small(MX + 0.08 + sw + 0.08, 3.52, sw, sh, BLUE_F, BLUE_E, [("Random forest", "bold"), ("500 trees", "normal")])
small(MX + 0.08, 2.22, MW - 0.16, 0.58, BLUE_F, BLUE_E, [("DBSCAN clustering", "bold"), ("radius 150 m, ≥ 60 incidents", "normal")])
small(MX + 0.08, 0.92, MW - 0.16, 0.58, BLUE_F, BLUE_E, [("Poisson per shift block", "bold"), ("5 blocks + hourly profile", "normal")])

# models -> evaluation
for r in (R1, R2, R3):
    arrow((MX + MW, mid(r)), (EX, mid(r)))

# ---------- evaluation column ----------
def ebox(r, title):
    box(EX, r[0], EW, r[1] - r[0], AMB_F, AMB_E)
    txt(EX + 0.12, r[1] - 0.20, title, 8.5, "bold", ha="left")
ebox(R1, "Busy-day forecast")
x0 = EX + 0.12
txt(x0, 5.63, f"Each of {n_test} test days predicted; error = MAE", 7.5, ha="left")
txt(x0, 5.27, "Model", 7.5, "bold", ha="left"); txt(EX + 1.85, 5.27, "MAE/day", 7.5, "bold", ha="right"); txt(EX + 2.63, 5.27, "% of mean", 7.5, "bold", ha="right")
line([x0, EX + EW - 0.12], [5.17, 5.17], color=AMB_E, lw=0.7)
rows = [("Baseline 1", "b1"), ("Baseline 2", "b2"), ("Poisson regression", "poi"), ("Random forest", "rf")]
for i, (name, k) in enumerate(rows):
    y = 4.94 - i * 0.24; wt = "bold" if k == "poi" else "normal"
    txt(x0, y, name, 7.5, wt, ha="left"); txt(EX + 1.85, y, f"{mae[k]:.2f}", 7.5, wt, ha="right"); txt(EX + 2.63, y, f"{mae[k] / mean * 100:.1f}%", 7.5, wt, ha="right")
line([x0, EX + EW - 0.12], [3.99, 3.99], color=AMB_E, lw=0.7)
txt(x0, 3.82, f"Mean = {mean:.1f} incidents/day; lower is better", 7, ha="left", color=MUTED, style="italic")
b1r, poir = round(mae['b1'], 2), round(mae['poi'], 2)   # as shown in Table 7
txt(x0, 3.62, f"Poisson is {(b1r - poir) / b1r * 100:.1f}% better than Baseline 1", 7.5, ha="left")

ebox(R2, "Hot-spot list and map")
for i, s in enumerate([f"• Incidents inside hot spots: {HS_TRAIN:.1f}% → {HS_TEST:.1f}%", f"   (randomly placed areas: {HS_RANDOM:.1f}%)",
                       f"• {TOP10_REAPPEAR} of the 10 largest reappear in 2025–2026", "• Persistence holds for all 9 settings tried"]):
    txt(x0, 2.89 - i * 0.18, s, 7.5, ha="left")

ebox(R3, "Staffing profile by time of day")
for i, s in enumerate(["• Hourly pattern matches across periods:", f"   r = {R_WEEKDAY:.2f} weekdays, {R_WEEKEND:.2f} weekends",
                       f"• Shift-block error: {blk_base:.2f} (baseline) vs", f"   {blk_poi:.2f} (Poisson) incidents per block"]):
    txt(x0, 1.59 - i * 0.18, s, 7.5, ha="left")

# held-out data -> evaluation (dashed amber bus)
yb = 0.22
line([LX + LW / 2, LX + LW / 2], [0.57, yb], color=AMB_E, lw=1.4, ls=(0, (4, 2)))
line([LX + LW / 2, BUS_TEST], [yb, yb], color=AMB_E, lw=1.4, ls=(0, (4, 2)))
line([BUS_TEST, BUS_TEST], [yb, mid(R1)], color=AMB_E, lw=1.4, ls=(0, (4, 2)))
for r in (R1, R2, R3):
    arrow((BUS_TEST, mid(r)), (EX + EW, mid(r)), color=AMB_E, lw=1.4, ls=(0, (4, 2)))
txt(4.3, 0.42, "held-out test data: used only to measure accuracy, never to fit a model", 7.5, color=AMB_T, style="italic")

fig.savefig("Methodology/workflow_figure.png", dpi=220)
print("saved", f"{mean:.2f}", {k: round(v, 2) for k, v in mae.items()})
