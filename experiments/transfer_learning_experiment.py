"""Transfer learning to a new chip type: treat MCU-C as new; MCU-A/B are the existing chips.
Usage (from the project root, with mcu_trim_fuse.csv in this folder or given as 2nd argument):
    python experiments/transfer_learning_experiment.py 0   # then 1, for the two test-wafer directions
Writes transfer_raw_<direction>.csv"""
import os, json, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd, warnings, time
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor as RF
from mcu_trim_utils import make_features, physics_estimate, evaluate_codes, TARGETS
DATA = sys.argv[2] if len(sys.argv) > 2 else "mcu_trim_fuse.csv"
card = json.load(open("models/model_card.json")); hp = card["hyperparameters"]
df = pd.read_csv(DATA)
base_df = df[df.derivative != "MCU-C"]; c = df[df.derivative == "MCU-C"]
waf = sorted(c.wafer_id.unique()); print("MCU-C wafers:", waf, [int((c.wafer_id == w).sum()) for w in waf], flush=True)
NT, W, DRAWS = 60, 20.0, 4
def params(k): p = hp[k]; return dict(min_samples_leaf=p["min_samples_leaf"], max_features=p["max_features"])
def fit_predict(train, test, key, weights=None):
    X = make_features(train).to_numpy(); b = physics_estimate(train)[key].to_numpy()
    m = RF(NT, random_state=0, n_jobs=1, **params(key)).fit(X, train[TARGETS[key]["column"]].to_numpy() - b, sample_weight=weights)
    return m.predict(make_features(test).to_numpy()) + physics_estimate(test)[key].to_numpy()
rows = []
t0 = time.time()
DIRS = [(waf[0], waf[1]), (waf[1], waf[0])]
D = int(sys.argv[1])
for direction, (tr_w, te_w) in [(D, DIRS[D])]:
    pool = c[c.wafer_id == tr_w].reset_index(drop=True); test = c[c.wafer_id == te_w].reset_index(drop=True)
    for key, spec in TARGETS.items():
        y = test[spec["column"]]
        rows.append(dict(setup="physics formula only", k=0, key=key, **evaluate_codes(y, physics_estimate(test)[key], spec["bits"])))
        rows.append(dict(setup="A+B only (zero-shot)", k=0, key=key, **evaluate_codes(y, fit_predict(base_df, test, key), spec["bits"])))
        for k in [10, 25, 50, 100, len(pool)]:
            for d in range(1 if k == len(pool) else DRAWS):
                samp = pool.sample(k, random_state=d) if k < len(pool) else pool
                tr = pd.concat([base_df, samp], ignore_index=True)
                w = np.r_[np.ones(len(base_df)), np.full(len(samp), W)]
                rows.append(dict(setup="A+B + k MCU-C dies (transfer)", k=k, key=key, **evaluate_codes(y, fit_predict(tr, test, key, w), spec["bits"])))
                if k < len(pool):
                    rows.append(dict(setup="k MCU-C dies only (from scratch)", k=k, key=key, **evaluate_codes(y, fit_predict(samp, test, key), spec["bits"])))
    print("direction", direction, "done", round(time.time() - t0), "s", flush=True)
r = pd.DataFrame(rows)
r.to_csv(f"transfer_raw_{D}.csv", index=False)
