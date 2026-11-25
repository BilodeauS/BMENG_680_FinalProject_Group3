"""Clean the City of Calgary Traffic Incidents file.

Usage: python3 clean_data.py RAW.csv OUT.csv

Steps (report Section 2.2):
 1. drop duplicate rows and rows with no incident description
 2. fill blank QUADRANT: (i) original label, (ii) NE/NW/SE/SW suffix of INCIDENT INFO,
    (iii) 5-nearest-neighbour classifier on latitude/longitude trained on the labelled rows.
    QUADRANT_SOURCE records which method was used.
 (the suffix match is case-sensitive, so lower-case endings such as "nE" fall through to step iii)
 3. convert START_DT_UTC to America/Edmonton local time; derive date, hour, weekday, month, year
"""
import re, sys
import numpy as np, pandas as pd
from sklearn.neighbors import KNeighborsClassifier

raw, out = sys.argv[1], sys.argv[2]
df = pd.read_csv(raw)
n0 = len(df)

# 1. duplicates and missing descriptions
df = df.drop_duplicates()
df = df[df["DESCRIPTION"].notna() & (df["DESCRIPTION"].astype(str).str.strip() != "")].reset_index(drop=True)

# 2. quadrant
q = df["QUADRANT"].astype("string").str.strip().str.upper()
q = q.where(q.isin(["NE", "NW", "SE", "SW"]))
src = pd.Series(np.where(q.notna(), "original", ""), index=df.index)

suffix = df["INCIDENT INFO"].astype("string").str.extract(r"\b(NE|NW|SE|SW)\s*$")[0]
use_text = q.isna() & suffix.notna()
q[use_text] = suffix[use_text]
src[use_text] = "from_incident_text"

todo = q.isna() & df["Latitude"].notna() & df["Longitude"].notna()
lab = q.notna() & (src == "original") & df["Latitude"].notna() & df["Longitude"].notna()
knn = KNeighborsClassifier(n_neighbors=5, algorithm="brute").fit(df.loc[lab, ["Latitude", "Longitude"]], q[lab].astype(str))
q[todo] = knn.predict(df.loc[todo, ["Latitude", "Longitude"]])
src[todo] = "from_coordinates"
df["QUADRANT"], df["QUADRANT_SOURCE"] = q, src

# 3. local time
t = pd.to_datetime(df["START_DT_UTC"], format="%Y/%m/%d %I:%M:%S %p", utc=True).dt.tz_convert("America/Edmonton")
df["START_LOCAL"] = t.dt.tz_localize(None).dt.strftime("%Y-%m-%d %H:%M:%S")
df["DATE_LOCAL"] = t.dt.strftime("%Y-%m-%d")
df["HOUR_LOCAL"] = t.dt.hour
df["DAY_OF_WEEK"] = t.dt.day_name()
df["MONTH"] = t.dt.month
df["YEAR"] = t.dt.year

df.to_csv(out, index=False)
print(f"{n0} raw rows -> {len(df)} cleaned rows")
print(df["QUADRANT_SOURCE"].value_counts().to_string())
print("still missing quadrant:", int(df["QUADRANT"].isna().sum()))
