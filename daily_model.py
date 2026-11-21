"""Daily incident-count forecasting for Calgary traffic incidents.
Train 2022-2024, test 2025-2026 (recording-gap days excluded). Compares a
historical-average baseline, Poisson regression and a random forest."""
import numpy as np, pandas as pd
from dateutil.easter import easter
from sklearn.linear_model import PoissonRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv("Traffic_Incidents_cleaned.csv", parse_dates=["DATE_LOCAL"])
d = df.groupby("DATE_LOCAL").size()
d = d.reindex(pd.date_range(d.index.min(), d.index.max()), fill_value=0).rename("count").to_frame()

# --- days to exclude: recording gaps found in the coverage check
gaps = [("2016-12-21","2017-02-07"),("2019-04-19","2019-04-28"),("2019-06-14","2019-09-15"),
        ("2025-05-21","2025-07-01")]
bad = pd.Series(False, index=d.index)
for a, b in gaps: bad |= (d.index >= a) & (d.index <= b)
for x in ["2019-03-14","2020-12-25","2025-01-20","2026-09-17"]: bad[pd.Timestamp(x)] = True
d = d[~bad]

# --- Alberta holidays (hand-coded; no external package needed)
def nth_weekday(y, m, wd, n):
    first = pd.Timestamp(y, m, 1); return first + pd.Timedelta(days=(wd-first.weekday())%7 + 7*(n-1))
hol = set()
for y in range(2016, 2027):
    vic = pd.Timestamp(y,5,24); vic -= pd.Timedelta(days=(vic.weekday()-0)%7)
    hol |= {pd.Timestamp(y,1,1), nth_weekday(y,2,0,3), pd.Timestamp(easter(y))-pd.Timedelta(days=2), vic,
            pd.Timestamp(y,7,1), nth_weekday(y,8,0,1), nth_weekday(y,9,0,1), nth_weekday(y,10,0,2),
            pd.Timestamp(y,11,11), pd.Timestamp(y,12,25), pd.Timestamp(y,12,26)}
d["dow"] = d.index.dayofweek; d["month"] = d.index.month
d["holiday"] = d.index.isin(hol).astype(int)

train = d[(d.index >= "2022-01-01") & (d.index <= "2024-12-31")]
test  = d[d.index >= "2025-01-01"]
print(f"train days {len(train)}, test days {len(test)}")

def design(x):
    return pd.concat([pd.get_dummies(x.dow, prefix="dow"), pd.get_dummies(x.month, prefix="mo"), x.holiday], axis=1).astype(float)
Xtr, Xte = design(train), design(test).reindex(columns=design(train).columns, fill_value=0)

res = {}
res["Baseline: day-of-week average"] = test.dow.map(train.groupby("dow")["count"].mean())
res["Baseline: day-of-week x month average"] = pd.Series([train[(train.dow==a)&(train.month==b)]["count"].mean() for a,b in zip(test.dow,test.month)], index=test.index)
res["Poisson regression"] = PoissonRegressor(alpha=1e-3, max_iter=1000).fit(Xtr, train["count"]).predict(Xte)
res["Random forest"] = RandomForestRegressor(n_estimators=500, min_samples_leaf=5, random_state=0).fit(Xtr, train["count"]).predict(Xte)

rows = []
for k, p in res.items():
    rows.append((k, mean_absolute_error(test["count"], p), np.sqrt(np.mean((test["count"]-p)**2))))
out = pd.DataFrame(rows, columns=["model","MAE","RMSE"]).round(2)
base = out.MAE.iloc[0]; out["MAE_vs_dow_baseline_%"] = ((base-out.MAE)/base*100).round(1)
print(out.to_string(index=False))
pd.DataFrame({"actual": test["count"], **{k: np.asarray(v) for k, v in res.items()}}).to_csv("daily_test_predictions.csv")
d.to_csv("daily_counts_model_table.csv")
