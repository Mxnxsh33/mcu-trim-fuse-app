"""
mcu_trim_utils.py
=================
Shared code for the MCU trim-code model. The SAME functions are used for
training (in the notebook) and for serving (backend / Streamlit later), so
there is no train/serve mismatch.

Quick use (backend):
    from mcu_trim_utils import TrimPredictor
    predictor = TrimPredictor.load("models/mcu_trim_model.joblib")
    result = predictor.predict({
        "derivative": "MCU-A",
        "bg_untrim_V": 1.19017, "hsi_untrim_MHz": 16.98648, "ldo_untrim_V": 1.09439,
        "pcm_vth_n_mV": 452.04, "pcm_vth_p_mV": -477.49, "pcm_rs_ohm_sq": 284.88,
        "pcm_tox_A": 20.435, "pcm_mobility_rel": 1.0356,
    })
"""
import numpy as np
import pandas as pd
import joblib

# ----------------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------------
DERIVATIVES = ["MCU-A", "MCU-B", "MCU-C"]                       # chip types known before trimming
UNTRIM_COLS = ["bg_untrim_V", "hsi_untrim_MHz", "ldo_untrim_V"]  # raw outputs measured BEFORE trimming
PCM_COLS = ["pcm_vth_n_mV", "pcm_vth_p_mV", "pcm_rs_ohm_sq",
            "pcm_tox_A", "pcm_mobility_rel"]                     # process-monitor readings
REQUIRED_INPUTS = UNTRIM_COLS + PCM_COLS + ["derivative"]        # what a user/backend must provide
FEATURE_NAMES = UNTRIM_COLS + PCM_COLS + ["is_MCU_B", "is_MCU_C"]  # what the model actually sees (order matters)

TARGETS = {                                                      # the three answers we predict
    "bg":  {"column": "bg_trim_code_opt",  "bits": 6},           # 6-bit bandgap code, 0..63
    "hsi": {"column": "hsi_trim_code_opt", "bits": 7},           # 7-bit HSI oscillator code, 0..127
    "ldo": {"column": "ldo_trim_code_opt", "bits": 5},           # 5-bit LDO code, 0..31
}


# ----------------------------------------------------------------------------
# Physics estimate (the "starting guess" the model corrects)
# ----------------------------------------------------------------------------
def physics_estimate(df):
    """Rough trim code from plain arithmetic, using the fuse-schema step sizes.
    bandgap: 2 mV/LSB, centre 32 -> target 1.200 V
    HSI    : 0.2 %/LSB, centre 64 -> target 16 MHz
    LDO    : 5 mV/LSB, centre 16 -> target 1.100 V
    """
    return pd.DataFrame({
        "bg":  32 + (1.200 - df["bg_untrim_V"]) / 0.002,
        "hsi": 64 + (16.0 / df["hsi_untrim_MHz"] - 1.0) / 0.002,
        "ldo": 16 + (1.100 - df["ldo_untrim_V"]) / 0.005,
    }, index=df.index)


def make_features(df):
    """Turn raw columns into the model's input table (same order every time)."""
    X = df[UNTRIM_COLS + PCM_COLS].astype(float).copy()
    X["is_MCU_B"] = (df["derivative"] == "MCU-B").astype(int)    # chip type as 0/1 flags
    X["is_MCU_C"] = (df["derivative"] == "MCU-C").astype(int)    # MCU-A is the "both flags 0" case
    return X[FEATURE_NAMES]


def clip_round(pred, bits):
    """Round to a whole code and keep it inside the valid bit range."""
    return np.clip(np.round(pred), 0, 2 ** bits - 1).astype(int)


def evaluate_codes(y_true, y_pred, bits):
    """Accuracy numbers for trim codes. 'ai_iterations' uses the same rule as the dataset's baseline:
    exact -> 1 check, off by 1 -> 2 checks, otherwise fall back to the full binary search (1 + bits)."""
    err = np.abs(clip_round(y_pred, bits) - np.asarray(y_true))
    ai = np.where(err == 0, 1, np.where(err <= 1, 2, 1 + bits)).mean()
    return {"exact_%": round(float((err == 0).mean() * 100), 2),
            "within_1_%": round(float((err <= 1).mean() * 100), 2),
            "mae_codes": round(float(err.mean()), 3),
            "ai_iterations": round(float(ai), 2)}


# ----------------------------------------------------------------------------
# Bundle + predictor (used by the backend)
# ----------------------------------------------------------------------------
def build_bundle(models, X_train, meta=None):
    """Package fitted models + everything needed to serve them into one dict."""
    ranges = {c: (float(X_train[c].min()), float(X_train[c].max()))
              for c in UNTRIM_COLS + PCM_COLS}                   # allowed input ranges seen in training
    return {"models": models, "feature_names": FEATURE_NAMES, "targets": TARGETS,
            "input_ranges": ranges, "derivatives": DERIVATIVES, "meta": meta or {}}


class TrimPredictor:
    """Loads the saved bundle and predicts trim codes for one die (dict) or many dies (DataFrame)."""

    def __init__(self, bundle):
        self.bundle = bundle
        self.models = bundle["models"]

    @classmethod
    def load(cls, path="models/mcu_trim_model.joblib"):
        return cls(joblib.load(path))

    def _to_frame(self, data):
        df = pd.DataFrame([data]) if isinstance(data, dict) else data.copy()
        missing = [c for c in REQUIRED_INPUTS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required inputs: {missing}")
        bad_deriv = set(df["derivative"]) - set(DERIVATIVES)
        if bad_deriv:
            raise ValueError(f"Unknown derivative {sorted(bad_deriv)}; allowed: {DERIVATIVES}")
        num = df[UNTRIM_COLS + PCM_COLS].apply(pd.to_numeric, errors="coerce")
        if num.isna().any().any():
            raise ValueError("Inputs must be numbers with no empty values.")
        if (num[UNTRIM_COLS] <= 0).any().any():
            raise ValueError("Untrimmed bandgap / HSI / LDO values must be positive.")
        return df

    def _warnings(self, df):
        """Flag inputs far outside the range seen in training (model is less trustworthy there)."""
        notes = [[] for _ in range(len(df))]
        for col, (lo, hi) in self.bundle["input_ranges"].items():
            margin = 0.10 * (hi - lo)
            for i, v in enumerate(df[col].astype(float).to_numpy()):
                if v < lo - margin or v > hi + margin:
                    notes[i].append(f"{col} outside training range")
        return ["; ".join(n) for n in notes]

    def predict(self, data):
        df = self._to_frame(data)
        X = make_features(df).to_numpy()
        base = physics_estimate(df)
        out = pd.DataFrame(index=df.index)
        for key, spec in TARGETS.items():
            rf = self.models[key]
            pred = rf.predict(X) + base[key].to_numpy()          # model learns the correction to the physics guess
            out[f"{key}_trim_code"] = clip_round(pred, spec["bits"])
            votes = np.stack([t.predict(X) for t in rf.estimators_])   # every tree's own opinion
            out[f"{key}_spread"] = votes.std(axis=0).round(3)    # disagreement between trees (indicative only)
        out["input_warnings"] = self._warnings(df)
        return out.reset_index(drop=True)
