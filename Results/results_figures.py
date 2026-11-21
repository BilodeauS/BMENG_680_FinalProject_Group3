"""Figures/tables for the Results section: daily forecast plot, training hot-spot map, top-10 hot spots with names."""
import numpy as np, pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import BallTree
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R=6371000; EPS=150; MN=60
df = pd.read_csv("Traffic_Incidents_cleaned.csv", parse_dates=["DATE_LOCAL"])
df = df[~((df.DATE_LOCAL>="2025-05-21")&(df.DATE_LOCAL<="2025-07-01")) & (df.YEAR>=2022)].reset_index(drop=True)
tr = df[df.YEAR<=2024].reset_index(drop=True); te = df[df.YEAR>=2025].reset_index(drop=True)
lab = DBSCAN(eps=EPS/R,min_samples=MN,metric="haversine",algorithm="ball_tree").fit_predict(np.radians(tr[["Latitude","Longitude"]].values))
clean = lambda s: s.str.replace(r"^(Northbound|Southbound|Eastbound|Westbound)\s+","",regex=True).mode().iat[0]
g = tr.assign(c=lab)[lab>=0].groupby("c").agg(train_n=("c","size"), lat=("Latitude","mean"), lon=("Longitude","mean"), quadrant=("QUADRANT",lambda s:s.mode().iat[0]), location=("INCIDENT INFO",clean)).sort_values("train_n",ascending=False)
core = tr[lab>=0]
d, idx = BallTree(np.radians(core[["Latitude","Longitude"]].values),metric="haversine").query(np.radians(te[["Latitude","Longitude"]].values),k=1)
near = d[:,0]*R<=EPS; cid = lab[lab>=0][idx[:,0]]
g["test_n"] = pd.Series(cid[near]).value_counts().reindex(g.index).fillna(0)
dt, de = tr.DATE_LOCAL.nunique(), te.DATE_LOCAL.nunique()
g["train_per_yr"] = (g.train_n/dt*365).round(0); g["test_per_yr"] = (g.test_n/de*365).round(0)
g = g.reset_index(drop=True); g.index += 1
print(g.head(10)[["location","quadrant","train_per_yr","test_per_yr"]].to_string())
g.to_csv("hotspots_train_named.csv")
# map: train hot spots sized by yearly rate, over all incidents
fig,ax=plt.subplots(figsize=(7.5,7.5))
ax.scatter(df.Longitude,df.Latitude,s=0.6,c="lightgrey")
sc=ax.scatter(g.lon,g.lat,s=g.train_per_yr*2.2,c=g.train_per_yr,cmap="Reds",edgecolor="k",linewidth=.4)
for i in range(1,6): ax.annotate(str(i),(g.lon[i],g.lat[i]),fontsize=9,fontweight="bold",xytext=(5,5),textcoords="offset points")
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude"); plt.colorbar(sc,label="Incidents per year, 2022-2024"); plt.tight_layout(); plt.savefig("hotspots_train_map.png",dpi=130)
# daily forecast figure
p = pd.read_csv("daily_test_predictions.csv",index_col=0,parse_dates=True)
full = pd.date_range(p.index.min(),p.index.max()); q = p.reindex(full)
fig,ax=plt.subplots(figsize=(10,3.8))
ax.plot(q.index,q.actual,c="0.55",lw=.8,label="Actual"); ax.plot(q.index,q["Poisson regression"],c="tab:blue",lw=1,label="Poisson regression")
ax.axvspan(pd.Timestamp("2025-05-21"),pd.Timestamp("2025-07-01"),color="red",alpha=.12,label="Excluded (recording gap)")
top = (p.actual-p["Poisson regression"]).abs().nlargest(3)
for dte in top.index: ax.annotate(dte.strftime("%d %b %Y"),(dte,p.actual[dte]),fontsize=8,xytext=(4,-3),textcoords="offset points")
ax.set_ylabel("Incidents per day"); ax.legend(loc="upper left",ncol=3,fontsize=8); plt.tight_layout(); plt.savefig("daily_forecast.png",dpi=130)
