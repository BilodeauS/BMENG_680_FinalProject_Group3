"""Validate DBSCAN hot spots: fit on 2022-2024, check they persist in 2025-2026."""
import numpy as np, pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree
from scipy.stats import spearmanr

R = 6371000
df = pd.read_csv("Cleaning/Traffic_Incidents_cleaned.csv", parse_dates=["DATE_LOCAL"])
gap = (df.DATE_LOCAL >= "2025-05-21") & (df.DATE_LOCAL <= "2025-07-01")   # recording outage
df = df[~gap & (df.YEAR >= 2022)].copy()
tr = df[df.YEAR <= 2024].reset_index(drop=True); te = df[df.YEAR >= 2025].reset_index(drop=True)
days = lambda x: x.DATE_LOCAL.nunique()
print(f"train {len(tr)} incidents / {days(tr)} days ; test {len(te)} incidents / {days(te)} days")

def rad(x): return np.radians(x[["Latitude","Longitude"]].values)
def fit(x, eps, mn):
    lab = DBSCAN(eps=eps/R, min_samples=mn, metric="haversine", algorithm="ball_tree").fit_predict(rad(x)); return lab

def centroids(x, lab):
    g = x.assign(c=lab)[lab>=0].groupby("c").agg(n=("c","size"), lat=("Latitude","mean"), lon=("Longitude","mean"))
    return g.sort_values("n", ascending=False)

def within(points, ref_xy, eps):
    """boolean: point lies within eps metres of any reference point"""
    d, _ = BallTree(np.radians(ref_xy), metric="haversine").query(np.radians(points), k=1); return d[:,0]*R <= eps

EPS, MN = 150, 60
lab = fit(tr, EPS, MN); cen = centroids(tr, lab)
core = tr[lab>=0][["Latitude","Longitude"]].values
print(f"\n== 1. Persistence of {len(cen)} hot spots found in 2022-2024 (eps={EPS} m, min={MN})")
tr_in = (lab>=0).mean()
te_in = within(te[["Latitude","Longitude"]].values, core, EPS).mean()
print(f"share of incidents inside hot spots: train {tr_in:.1%} -> test {te_in:.1%}")
# per-day rate comparison
rate_tr = (lab>=0).sum()/days(tr); rate_te = within(te[["Latitude","Longitude"]].values, core, EPS).sum()/days(te)
print(f"hot-spot incidents/day: train {rate_tr:.2f} -> test {rate_te:.2f}")

# area control: the same circles shifted to random incident locations
rng = np.random.default_rng(0); allxy = df[["Latitude","Longitude"]].values
ctrl = []
for _ in range(20):
    # keep cluster sizes/footprint: place each hot-spot's core points at an offset to a random incident location
    off = allxy[rng.integers(len(allxy))] - cen[["lat","lon"]].mean().values
    shifted = core + off; ctrl.append(within(te[["Latitude","Longitude"]].values, shifted, EPS).mean())
print(f"control (same footprint shifted to random spots): {np.mean(ctrl):.1%} of test incidents")

print("\n== 2. Per-hot-spot counts, train vs test (incidents per 365 days)")
d, idx = BallTree(np.radians(core), metric="haversine").query(np.radians(te[["Latitude","Longitude"]].values), k=1)
near = d[:,0]*R <= EPS
cid = lab[lab>=0][idx[:,0]]
tec = pd.Series(cid[near]).value_counts()
cmp = cen.assign(test=tec.reindex(cen.index).fillna(0))
cmp["train_per_yr"] = cmp.n/days(tr)*365; cmp["test_per_yr"] = cmp.test/days(te)*365
rho = spearmanr(cmp.train_per_yr, cmp.test_per_yr).correlation
print(f"Spearman rank correlation train vs test: {rho:.2f}")
print(f"hot spots with test rate >= half their train rate: {(cmp.test_per_yr >= .5*cmp.train_per_yr).mean():.0%}")
top10 = cmp.head(10); print(top10[["train_per_yr","test_per_yr"]].round(0).astype(int).to_string())

print("\n== 3. Top-10 overlap when clustering test period independently")
mn_te = round(MN*days(te)/days(tr))   # same incidents-per-day density as the training fit
labt = fit(te, EPS, mn_te); cent = centroids(te, labt)
print(f"test clustering uses min_samples={mn_te} (same density threshold); finds {len(cent)} hot spots")
def top_match(a, b, k=10, tol=300):
    A = a.head(k)[["lat","lon"]].values; B = b.head(k)[["lat","lon"]].values
    return int(within(A, B, tol).sum())
print(f"top-10 train hot spots also in test top-10 (within 300 m): {top_match(cen, cent)}/10")
print(f"top-10 train hot spots also somewhere in test hot spots: {int(within(cen.head(10)[['lat','lon']].values, cent[['lat','lon']].values, 300).sum())}/10")

print("\n== 4. Year-to-year top-10 stability (clustering each year separately)")
yrs = {}
for y in [2022,2023,2024,2025]:
    x = df[df.YEAR==y].reset_index(drop=True); yrs[y] = centroids(x, fit(x, EPS, round(MN*days(x)/days(tr))))   # same density threshold per year
for a,b in [(2022,2023),(2023,2024),(2024,2025)]:
    print(f"{a}->{b}: {top_match(yrs[a], yrs[b])}/10 of top-10 recur")

print("\n== 5. Sensitivity to parameters (train fit -> share of test incidents inside; top-10 overlap w/ default)")
rows = []
for eps in [100,150,250]:
    for mn in [40,60,100]:
        l = fit(tr, eps, mn); c = centroids(tr, l)
        if len(c)==0: continue
        cp = tr[l>=0][["Latitude","Longitude"]].values
        rows.append((eps, mn, len(c), (l>=0).mean(), within(te[["Latitude","Longitude"]].values, cp, eps).mean(), top_match(cen, c) ))
s = pd.DataFrame(rows, columns=["eps_m","min_samples","n_hotspots","train_share","test_share","top10_overlap_default"])
print(s.round(3).to_string(index=False))
cmp.round(2).to_csv("Validations/hotspots_validation.csv"); s.round(3).to_csv("Validations/hotspots_sensitivity.csv", index=False)
