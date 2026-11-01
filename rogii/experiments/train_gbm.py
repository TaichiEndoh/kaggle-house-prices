import sys, numpy as np, pandas as pd, lightgbm as lgb
feat = pd.read_pickle(sys.argv[1])
drop = ["row_ev","y","well","fold","cst"]
cols = [c for c in feat.columns if c not in drop]
oof = np.zeros(len(feat), dtype=np.float32)
params = dict(objective="regression", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=4)
for f in range(5):
    tr = feat[(feat.fold != f) & (feat.row_ev % 5 == 0)]
    va = feat.fold == f
    m = lgb.train(params, lgb.Dataset(tr[cols], tr.y), num_boost_round=600)
    oof[va.values] = m.predict(feat.loc[va, cols])
    print("fold", f, "RMSE", np.sqrt(np.mean((oof[va.values]-feat.y[va].values)**2)), flush=True)
y = feat.y.values
print(f"const       RMSE = {np.sqrt(np.mean(y**2)):.3f}")
for c in ["plane3000","plane5000","plane8000"]:
    p = np.nan_to_num(feat[c].values); print(f"0.5*{c:9s} RMSE = {np.sqrt(np.mean((0.5*p-y)**2)):.3f}")
print(f"LightGBM    RMSE = {np.sqrt(np.mean((oof-y)**2)):.3f}")
imp = pd.Series(m.feature_importance("gain"), index=cols).sort_values(ascending=False)
print((imp/imp.sum()).head(10).round(3).to_string())
np.save(sys.argv[2], oof)
