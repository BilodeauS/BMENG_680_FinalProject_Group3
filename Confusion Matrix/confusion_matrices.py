"""Confusion matrices: (A) busy-day classification derived from forecasts, (B) quadrant tagger (kNN)."""
import numpy as np, pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

p = pd.read_csv("Forecasting/daily_test_predictions.csv", index_col=0, parse_dates=True)
t = pd.read_csv("Forecasting/daily_counts_model_table.csv", index_col=0, parse_dates=True)
tr = t[(t.index >= "2022-01-01") & (t.index <= "2024-12-31")]["count"]
thr = tr.quantile(0.75)                      # "busy" = at or above the training 75th percentile
busy = p.actual >= thr
models = {"Day-of-week baseline": "Baseline: day-of-week average", "Poisson regression": "Poisson regression", "Random forest": "Random forest"}

def mats(flag):
    out = {}
    for lab, col in models.items():
        yh = flag(p[col]); out[lab] = confusion_matrix(busy, yh, labels=[True, False])   # rows actual busy/not, cols predicted busy/not
    return out
A = mats(lambda s: s >= thr)                                        # fixed threshold
B = mats(lambda s: s.rank(pct=True, method="first") > 0.75)         # top 25% of days by prediction (exactly 25% flagged)
def stats(m): tp, fn, fp, tn = m[0,0], m[0,1], m[1,0], m[1,1]; return dict(TP=tp,FN=fn,FP=fp,TN=tn, accuracy=(tp+tn)/m.sum(), precision=tp/max(tp+fp,1), recall=tp/max(tp+fn,1))
rows = []
for nm, M in [(f"A: predicted >= {thr:.0f}", A), ("B: top 25% of days by prediction", B)]:
    for lab, m in M.items(): rows.append({"rule": nm, "model": lab, **stats(m)})
res = pd.DataFrame(rows); print(res.round(3).to_string(index=False)); res.round(3).to_csv("Confusion Matrix/confusion_busy_day_summary.csv", index=False)
print(f"threshold {thr:.0f}; busy test days {busy.sum()} of {len(p)} ({busy.mean():.1%}); busy train days {(tr>=thr).mean():.1%}")

# quadrant tagger: 5-fold cross-validated predictions on originally labelled incidents
df = pd.read_csv("Cleaning/Traffic_Incidents_cleaned.csv")
lab = df[df.QUADRANT_SOURCE == "original"]
yq = lab.QUADRANT.values
pred = cross_val_predict(KNeighborsClassifier(n_neighbors=5), lab[["Longitude","Latitude"]].values, yq, cv=5)
order = ["NE","NW","SW","SE"]; Q = confusion_matrix(yq, pred, labels=order)
print("quadrant accuracy %.3f" % (np.trace(Q)/Q.sum())); print(pd.DataFrame(Q, index=order, columns=order))
pd.DataFrame(Q, index=order, columns=order).to_csv("Confusion Matrix/confusion_quadrant.csv")

fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
def draw(a, m, xl, yl, title, rowtot=True):
    pct = m / m.sum(axis=1, keepdims=True); a.imshow(pct, cmap="Blues", vmin=0, vmax=1)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]): a.text(j, i, f"{m[i,j]:,}\n({pct[i,j]:.0%})", ha="center", va="center", color="white" if pct[i,j] > .5 else "black", fontsize=9)
    a.set_xticks(range(len(xl))); a.set_xticklabels(xl); a.set_yticks(range(len(yl))); a.set_yticklabels(yl); a.set_title(title, fontsize=10)
    a.set_xlabel("Predicted"); a.set_ylabel("Actual")
draw(ax[0], A["Poisson regression"], ["Busy", "Not busy"], ["Busy", "Not busy"], f"Poisson, rule A\n(predicted >= {thr:.0f} incidents)")
draw(ax[1], B["Poisson regression"], ["Busy", "Not busy"], ["Busy", "Not busy"], "Poisson, rule B\n(top 25% of days by prediction)")
draw(ax[2], Q, order, order, "Quadrant tagger (kNN, 5-fold CV)")
plt.tight_layout(); plt.savefig("Confusion Matrix/confusion_matrices.png", dpi=130)
