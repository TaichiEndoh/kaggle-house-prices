import glob, os, sys, numpy as np, pandas as pd
D = sys.argv[1]; OUT = sys.argv[2]
frames = []
for f in sorted(glob.glob(f"{D}/train/*__horizontal_well.csv")):
    w = os.path.basename(f).split("__")[0]
    h = pd.read_csv(f)
    h.insert(0, "row", np.arange(len(h)))
    h.insert(0, "well", w)
    frames.append(h)
df = pd.concat(frames, ignore_index=True)
for c in df.columns:
    if df[c].dtype == "float64": df[c] = df[c].astype("float64")
df.to_pickle(OUT)
print(df.shape, df.well.nunique(), "wells")
print("eval rows:", df.TVT_input.isna().sum(), " / total", len(df))
