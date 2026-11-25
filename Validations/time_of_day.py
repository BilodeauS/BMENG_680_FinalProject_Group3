"""Time-of-day analysis: hourly profile, shift-block forecasting, and hot spots by block.
Train 2022-2024, test 2025-2026 (recording-gap days excluded)."""
import numpy as np, pandas as pd
from dateutil.easter import easter
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.cluster import DBSCAN
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

df = pd.read_csv("Cleaning/Traffic_Incidents_cleaned.csv", parse_dates=["DATE_LOCAL"])
bad_ranges = [("2025-05-21","2025-07-01")]; bad_days = ["2026-09-17","2025-01-20"]
bad = pd.Series(False, index=df.index)
for a,b in bad_ranges: bad |= (df.DATE_LOCAL>=a)&(df.DATE_LOCAL<=b)
bad |= df.DATE_LOCAL.isin(pd.to_datetime(bad_days))
df = df[~bad & (df.YEAR>=2022)].copy()
BLOCKS = [(0,6,"Night 00-06"),(6,10,"AM rush 06-10"),(10,15,"Midday 10-15"),(15,19,"PM rush 15-19"),(19,24,"Evening 19-24")]
def block(h):
    for a,b,n in BLOCKS:
        if a<=h<b: return n
df["block"] = df.HOUR_LOCAL.map(block)
df["weekend"] = df.DATE_LOCAL.dt.dayofweek>=5
tr = df[df.YEAR<=2024]; te = df[df.YEAR>=2025]
ndays = lambda x: x.DATE_LOCAL.nunique()

# ---- 1. hourly profile (incidents per day), weekday vs weekend, train vs test
def prof(x, we):
    s = x[x.weekend==we]; nd = s.DATE_LOCAL.nunique(); return s.groupby("HOUR_LOCAL").size().reindex(range(24), fill_value=0)/nd
P = {(n,we): prof(x,we) for n,x in [("train",tr),("test",te)] for we in [False,True]}
for we,lab in [(False,"weekday"),(True,"weekend")]:
    r = np.corrcoef(P[("train",we)], P[("test",we)])[0,1]
    print(f"hourly profile {lab}: train-vs-test correlation {r:.3f}")
wd = P[("train",False)]; we_ = P[("train",True)]
print("weekday peak hour", int(wd.idxmax()), f"({wd.max():.2f}/h);  weekend peak hour", int(we_.idxmax()), f"({we_.max():.2f}/h);  weekday overnight (00-05) avg {wd[:6].mean():.2f}/h")
fig,ax=plt.subplots(figsize=(8,4.5))
for we,c,l in [(False,"tab:blue","Weekday"),(True,"tab:orange","Weekend")]:
    ax.plot(P[("train",we)].index,P[("train",we)].values,c=c,label=f"{l} 2022-24"); ax.plot(P[("test",we)].index,P[("test",we)].values,c=c,ls="--",label=f"{l} 2025-26")
ax.set_xlabel("Hour of day (Calgary local time)"); ax.set_ylabel("Incidents per hour per day"); ax.set_xticks(range(0,24,3)); ax.legend(); ax.set_title("Incidents by hour: weekday vs weekend")
plt.tight_layout(); plt.savefig("Validations/hourly_profile.png",dpi=110)

# ---- 2. forecast incidents per day x block
def table(x):
    days = pd.date_range(x.DATE_LOCAL.min(), x.DATE_LOCAL.max()); idx = pd.MultiIndex.from_product([days,[b[2] for b in BLOCKS]],names=["date","block"])
    t = x.groupby(["DATE_LOCAL","block"]).size().reindex(idx, fill_value=0).rename("count").reset_index(); return t
def hol(y):
    def nth(y,m,wd,n): f=pd.Timestamp(y,m,1); return f+pd.Timedelta(days=(wd-f.weekday())%7+7*(n-1))
    vic=pd.Timestamp(y,5,24); vic-=pd.Timedelta(days=vic.weekday())
    return {pd.Timestamp(y,1,1),nth(y,2,0,3),pd.Timestamp(easter(y))-pd.Timedelta(days=2),vic,pd.Timestamp(y,7,1),nth(y,8,0,1),nth(y,9,0,1),nth(y,10,0,2),pd.Timestamp(y,11,11),pd.Timestamp(y,12,25),pd.Timestamp(y,12,26)}
H = set().union(*[hol(y) for y in range(2022,2027)])
valid = set(df.DATE_LOCAL.unique())   # days that have data (gap days removed)
T = table(df); T = T[T.date.isin(valid)]
T["dow"]=T.date.dt.dayofweek; T["month"]=T.date.dt.month; T["holiday"]=T.date.isin(H).astype(int); T["year"]=T.date.dt.year
Ttr, Tte = T[T.year<=2024], T[T.year>=2025]
base_dow = Ttr.groupby(["block","dow"])["count"].mean(); base_blk = Ttr.groupby("block")["count"].mean()
pred = pd.DataFrame(index=Tte.index)
pred["Baseline: block average"] = Tte.block.map(base_blk)
pred["Baseline: block x day-of-week avg"] = [base_dow[(b,d)] for b,d in zip(Tte.block,Tte.dow)]
pred["Poisson regression"] = 0.0
for b in Ttr.block.unique():
    a, c = Ttr[Ttr.block==b], Tte[Tte.block==b]
    des = lambda z: pd.concat([pd.get_dummies(z.dow,prefix="d"),pd.get_dummies(z.month,prefix="m"),z.holiday],axis=1).astype(float)
    Xa = des(a); Xc = des(c).reindex(columns=Xa.columns, fill_value=0)
    pred.loc[c.index,"Poisson regression"] = PoissonRegressor(alpha=1e-3,max_iter=1000).fit(Xa,a["count"]).predict(Xc)
rows=[]
for k in pred: rows.append((k, mean_absolute_error(Tte["count"],pred[k])))
res = pd.DataFrame(rows,columns=["model","MAE (incidents per block)"]); b0=res.iloc[1,1]
res["vs block x dow baseline %"] = ((b0-res.iloc[:,1])/b0*100).round(1)
print("\nforecast of incidents per day x shift block (test 2025-26)"); print(res.round(2).to_string(index=False))
per = pd.DataFrame({"block":Tte.block,"actual":Tte["count"],"base":pred["Baseline: block x day-of-week avg"],"poisson":pred["Poisson regression"]})
pb = per.groupby("block").apply(lambda g: pd.Series({"mean_actual":g.actual.mean(),"MAE_base":mean_absolute_error(g.actual,g.base),"MAE_poisson":mean_absolute_error(g.actual,g.poisson)})).reindex([b[2] for b in BLOCKS])
print("\nper block:"); print(pb.round(2).to_string())
pb.round(3).to_csv("Validations/time_block_forecast_by_block.csv"); res.round(3).to_csv("Validations/time_block_forecast_summary.csv",index=False)

# ---- 3. hot spots by time block (train period), eps 150 m, min 30 incidents
R=6371000; print("\nhot spots by block, 2022-2024 (eps 150 m, min_samples 30)")
allx = tr; out=[]
for a,b,n in BLOCKS:
    x = allx[allx.block==n].reset_index(drop=True)
    lab = DBSCAN(eps=150/R,min_samples=30,metric="haversine",algorithm="ball_tree").fit_predict(np.radians(x[["Latitude","Longitude"]].values))
    g = x.assign(c=lab)[lab>=0].groupby("c").agg(n=("c","size"),loc=("INCIDENT INFO",lambda s:s.str.replace(r"^(Northbound|Southbound|Eastbound|Westbound)\s+","",regex=True).mode().iat[0])).sort_values("n",ascending=False)
    print(f"{n:15s} {len(x):6d} incidents, {len(g):2d} hot spots ({(lab>=0).mean():.0%} of block's incidents); top 3: " + "; ".join(g['loc'].head(3)))
    for i,(c,r) in enumerate(g.head(5).iterrows(),1): out.append((n,i,r["loc"],r["n"]))
pd.DataFrame(out,columns=["block","rank","location","incidents_2022_24"]).to_csv("Validations/hotspots_by_time_block.csv",index=False)
