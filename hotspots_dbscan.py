"""DBSCAN hot-spot detection on incident coordinates (2022 onward)."""
import numpy as np, pandas as pd
from sklearn.cluster import DBSCAN
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

df = pd.read_csv("Traffic_Incidents_cleaned.csv")
df = df[df.YEAR >= 2022].reset_index(drop=True)
EPS_M, MIN_N = 150, 60
X = np.radians(df[["Latitude","Longitude"]].values)
lab = DBSCAN(eps=EPS_M/6371000, min_samples=MIN_N, metric="haversine", algorithm="ball_tree").fit_predict(X)
df["cluster"] = lab
hs = df[df.cluster>=0].groupby("cluster").agg(incidents=("cluster","size"), lat=("Latitude","mean"), lon=("Longitude","mean"),
        quadrant=("QUADRANT", lambda s: s.mode().iat[0]),
        location=("INCIDENT INFO", lambda s: s.str.replace(r"^(Northbound|Southbound|Eastbound|Westbound)\s+","",regex=True).mode().iat[0])
      ).sort_values("incidents", ascending=False).reset_index(drop=True)
hs.index += 1
print(f"{len(df)} incidents, {len(hs)} hot spots, {(lab>=0).mean():.1%} of incidents fall in a hot spot (eps={EPS_M} m, min={MIN_N})")
print(hs.head(10).round(4).to_string())
hs.to_csv("hotspots_top.csv")
fig, ax = plt.subplots(figsize=(8,8))
ax.scatter(df.Longitude[lab<0], df.Latitude[lab<0], s=1, c="lightgrey", label="other incidents")
sc = ax.scatter(hs.lon, hs.lat, s=hs.incidents/4, c=hs.incidents, cmap="Reds", edgecolor="k", linewidth=.3)
ax.set_title(f"Incident hot spots, Calgary 2022 onward (DBSCAN, {EPS_M} m, >= {MIN_N} incidents)")
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); plt.colorbar(sc, label="incidents in hot spot"); plt.tight_layout()
plt.savefig("hotspots_map.png", dpi=110)
