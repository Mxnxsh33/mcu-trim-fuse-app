"""
MCU Trim-Code Predictor  (Streamlit app, styled UI)

Run locally:   python -m streamlit run app.py
The app loads the saved model and uses the SAME TrimPredictor class as training,
so predictions here match the notebook exactly.
"""
import json
from pathlib import Path

import pandas as pd
import streamlit as st

import assistant
import explain
import fuse_tools as ft
from mcu_trim_utils import (TrimPredictor, TARGETS, UNTRIM_COLS, PCM_COLS,
                            DERIVATIVES, physics_estimate, evaluate_codes)

APP_DIR = Path(__file__).parent
COLS = UNTRIM_COLS + PCM_COLS
NONE_CHOICE = "(none - enter my own values)"

st.set_page_config(page_title="MCU Trim Code Predictor", page_icon="🔧", layout="wide")

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;800&display=swap');
html, body, .stApp, [class*="st-"] { font-family: 'Inter', 'Segoe UI', sans-serif; }
/* keep Streamlit's icon font for arrows/icons (otherwise they show as text like "arrow_right") */
[data-testid="stIconMaterial"], span[class*="material"], .material-icons, .material-symbols-rounded {
  font-family: 'Material Symbols Rounded', 'Material Icons' !important;
}

[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(1100px 520px at 8% -8%, rgba(0,212,255,.20), transparent 60%),
    radial-gradient(900px 480px at 100% 0%, rgba(139,92,246,.24), transparent 60%),
    radial-gradient(800px 500px at 50% 115%, rgba(236,72,153,.14), transparent 60%),
    #0a0f1f;
}
[data-testid="stHeader"] { background: transparent; }
footer { visibility: hidden; }
.block-container { padding-top: 1.6rem; max-width: 1280px; }

/* ---------- hero ---------- */
.hero {
  position: relative; overflow: hidden; padding: 30px 34px 26px; margin-bottom: 22px;
  border-radius: 24px; border: 1px solid rgba(255,255,255,.14);
  background:
    linear-gradient(rgba(255,255,255,.035) 1px, transparent 1px) 0 0 / 34px 34px,
    linear-gradient(90deg, rgba(255,255,255,.035) 1px, transparent 1px) 0 0 / 34px 34px,
    linear-gradient(135deg, rgba(0,212,255,.12), rgba(139,92,246,.14));
  box-shadow: 0 18px 50px rgba(0,0,0,.45);
}
.hero::after {
  content: "⚙"; position: absolute; right: 26px; top: -18px; font-size: 150px;
  opacity: .07; transform: rotate(14deg);
}
.hero-badge {
  display: inline-block; padding: 5px 14px; border-radius: 999px; font-size: .78rem; font-weight: 600;
  letter-spacing: .06em; text-transform: uppercase; color: #67e8f9;
  background: rgba(0,212,255,.12); border: 1px solid rgba(0,212,255,.4);
}
.hero-title { font-size: 2.9rem; font-weight: 800; line-height: 1.08; margin: 12px 0 8px; color: #f3f6ff; }
.hero-title span {
  background: linear-gradient(90deg, #00d4ff, #8b5cf6 55%, #ec4899);
  -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
}
.hero-sub { color: #a9b6d8; font-size: 1.02rem; max-width: 760px; margin-bottom: 16px; }
.chips { display: flex; gap: 10px; flex-wrap: wrap; }
.chip {
  padding: 8px 15px; border-radius: 999px; font-size: .85rem; color: #dbe4ff;
  background: rgba(255,255,255,.07); border: 1px solid rgba(255,255,255,.16);
}
.chip b { color: #67e8f9; }

/* ---------- section titles ---------- */
.sec { display: flex; align-items: center; gap: 10px; font-size: 1.15rem; font-weight: 700; color: #f3f6ff; margin: 4px 0 12px; }
.sec .dot { width: 10px; height: 26px; border-radius: 6px; background: linear-gradient(180deg, #00d4ff, #8b5cf6); }
.mini { font-size: .8rem; letter-spacing: .08em; text-transform: uppercase; font-weight: 600; color: #8da0cc; margin: 14px 0 4px; }

/* ---------- glass containers ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {
  background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.12) !important;
  border-radius: 20px !important; backdrop-filter: blur(6px);
}

/* ---------- tabs as pills ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 10px; border-bottom: none; }
.stTabs [data-baseweb="tab"] {
  height: auto; padding: 9px 20px; border-radius: 999px; background: rgba(255,255,255,.06);
  border: 1px solid rgba(255,255,255,.12); color: #c3cff0; font-weight: 600;
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(90deg, rgba(0,212,255,.25), rgba(139,92,246,.28));
  border-color: rgba(0,212,255,.65); color: #fff;
}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display: none; }

/* ---------- inputs & buttons ---------- */
[data-baseweb="input"], [data-baseweb="select"] > div { background: rgba(255,255,255,.06) !important; border-radius: 10px !important; }
.stButton > button, .stDownloadButton > button {
  background: linear-gradient(90deg, #00b4e6, #7c3aed); color: #fff; border: 0; border-radius: 12px;
  font-weight: 600; padding: .55rem 1.1rem; transition: transform .15s, box-shadow .15s;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  transform: translateY(-2px); box-shadow: 0 8px 22px rgba(124,58,237,.45); color: #fff; border: 0;
}
[data-testid="stExpander"] {
  border-radius: 14px; background: rgba(255,255,255,.04); border: 1px solid rgba(255,255,255,.12);
}
[data-testid="stFileUploaderDropzone"] {
  border: 2px dashed rgba(0,212,255,.55); border-radius: 16px; background: rgba(0,212,255,.05);
}

/* ---------- result cards ---------- */
.tcard {
  position: relative; overflow: hidden; padding: 20px 20px 16px; border-radius: 20px; height: 100%;
  background: linear-gradient(160deg, rgba(255,255,255,.09), rgba(255,255,255,.02));
  border: 1px solid rgba(255,255,255,.14); box-shadow: 0 12px 32px rgba(0,0,0,.38);
  transition: transform .2s, box-shadow .2s;
}
.tcard::before { content: ""; position: absolute; left: 0; right: 0; top: 0; height: 5px; background: var(--accent); }
.tcard:hover { transform: translateY(-4px); box-shadow: 0 18px 40px rgba(0,0,0,.5); }
.tcard-title { font-size: .74rem; letter-spacing: .09em; text-transform: uppercase; color: #9fb0d6; font-weight: 600; }
.tcard-value {
  font-size: 3.8rem; font-weight: 800; line-height: 1.05; margin: 6px 0 2px;
  background: var(--accent); -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
}
.tcard-sub { font-size: .8rem; color: #8da0cc; }
.tcard-note { margin-top: 10px; font-size: .8rem; color: #c9d4f2; padding-top: 8px; border-top: 1px dashed rgba(255,255,255,.16); }

/* ---------- confidence ---------- */
.conf { border-radius: 14px; padding: 13px 16px; margin-top: 16px; border: 1px solid; font-size: .93rem; color: #eef2ff; }
.conf-high { background: rgba(34,197,94,.12); border-color: rgba(34,197,94,.5); }
.conf-med  { background: rgba(245,158,11,.12); border-color: rgba(245,158,11,.5); }
.conf-low  { background: rgba(239,68,68,.12); border-color: rgba(239,68,68,.5); }
.pill { display: inline-block; padding: 3px 12px; border-radius: 999px; font-weight: 700; font-size: .8rem; margin-right: 8px; color: #0a0f1f; }
.pill-high { background: #4ade80; } .pill-med { background: #fbbf24; } .pill-low { background: #f87171; }
.meter-wrap { position: relative; margin: 12px 0 4px; }
.meter { display: flex; height: 10px; border-radius: 99px; overflow: hidden; }
.pin { position: absolute; top: -4px; width: 4px; height: 18px; background: #fff; border-radius: 2px; box-shadow: 0 0 10px #fff; }
.meter-labels { display: flex; justify-content: space-between; font-size: .7rem; color: #8da0cc; }

/* ---------- about steps ---------- */
.steps { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin: 6px 0 16px; }
.step { padding: 18px; border-radius: 18px; background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.12); }
.step .n {
  display: inline-flex; width: 30px; height: 30px; border-radius: 50%; align-items: center; justify-content: center;
  font-weight: 800; color: #0a0f1f; background: linear-gradient(135deg, #00d4ff, #8b5cf6); margin-bottom: 8px;
}
.step b { color: #f3f6ff; } .step div.t { color: #a9b6d8; font-size: .9rem; margin-top: 4px; }

/* ---------- chat ---------- */
[data-testid="stChatMessage"] { background: rgba(255,255,255,.05); border: 1px solid rgba(255,255,255,.10); border-radius: 16px; padding: 12px 16px; margin-bottom: 10px; }
.mode { display: inline-block; margin-left: 12px; padding: 3px 12px; border-radius: 999px; font-size: .74rem; font-weight: 700; letter-spacing: .05em; vertical-align: middle; }
.mode-llm { background: rgba(34,197,94,.18); color: #4ade80; border: 1px solid rgba(34,197,94,.5); }
.mode-off { background: rgba(245,158,11,.15); color: #fbbf24; border: 1px solid rgba(245,158,11,.45); }

@media (max-width: 800px) {
  .hero-title { font-size: 2.1rem; } .steps { grid-template-columns: 1fr; } .tcard-value { font-size: 3rem; }
}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def html(s):
    """Flatten HTML to one line (indented/blank lines would be read as Markdown code)."""
    st.markdown(" ".join(line.strip() for line in s.splitlines() if line.strip()), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Load model + config once (cached across reruns)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_all():
    predictor = TrimPredictor.load(APP_DIR / "models" / "mcu_trim_model.joblib")
    card = json.loads((APP_DIR / "models" / "model_card.json").read_text())
    cfg = json.loads((APP_DIR / "models" / "app_config.json").read_text())
    samples = pd.read_csv(APP_DIR / "sample_dies.csv")
    return predictor, card, cfg, samples


predictor, card, cfg, samples = load_all()

# label, help text, min, max, step, display format
INPUTS = {
    "bg_untrim_V":      ("Bandgap before trim (V)", "Bandgap reference voltage measured before trimming. Target is 1.200 V.", 0.5, 2.0, 0.001, "%.5f"),
    "hsi_untrim_MHz":   ("HSI oscillator before trim (MHz)", "Internal oscillator frequency before trimming. Target is 16 MHz.", 5.0, 40.0, 0.01, "%.4f"),
    "ldo_untrim_V":     ("LDO output before trim (V)", "Voltage-regulator output before trimming. Target is 1.100 V.", 0.5, 2.0, 0.001, "%.5f"),
    "pcm_vth_n_mV":     ("NMOS threshold voltage (mV)", "Process-monitor reading from the wafer test structures.", 100.0, 900.0, 1.0, "%.2f"),
    "pcm_vth_p_mV":     ("PMOS threshold voltage (mV)", "Process-monitor reading (negative for PMOS).", -900.0, -100.0, 1.0, "%.2f"),
    "pcm_rs_ohm_sq":    ("Sheet resistance (ohm/sq)", "Process-monitor reading.", 50.0, 800.0, 1.0, "%.2f"),
    "pcm_tox_A":        ("Oxide thickness (angstrom)", "Process-monitor reading. Matters a lot for the HSI prediction.", 5.0, 40.0, 0.01, "%.3f"),
    "pcm_mobility_rel": ("Relative carrier mobility", "Process-monitor reading, 1.0 = nominal.", 0.3, 2.0, 0.001, "%.4f"),
}
TARGET_LABELS = {"bg": "Bandgap trim code", "hsi": "HSI oscillator trim code", "ldo": "LDO trim code"}
ACCENTS = {"bg": "linear-gradient(90deg,#00d4ff,#3b82f6)",
           "hsi": "linear-gradient(90deg,#8b5cf6,#ec4899)",
           "ldo": "linear-gradient(90deg,#10b981,#06b6d4)"}
CODE_RANGE = {k: f"0 to {2 ** v['bits'] - 1}" for k, v in TARGETS.items()}


def acc_text(key):
    """Accuracy statement from the two independent tests (held-out wafers, brand-new dataset)."""
    a = card["holdout_results"][key]["within_1_%"]
    b = cfg["fresh_data_check"]["results"][key]["within_1_%"]
    lo, hi = sorted([a, b])
    return f"{lo:.1f}% within ±1 code" if hi - lo < 0.5 else f"{lo:.0f}–{hi:.0f}% within ±1 code"


# ---------------------------------------------------------------------------
# Session state + callbacks (keep inputs in sync with chip type / examples)
# ---------------------------------------------------------------------------
def set_defaults(deriv):
    for c in COLS:
        st.session_state[f"in_{c}"] = float(cfg["defaults"][deriv][c])


def on_derivative_change():
    set_defaults(st.session_state["derivative"])
    st.session_state["example_choice"] = NONE_CHOICE
    st.session_state["example_truth"] = None


def on_example_change():
    choice = st.session_state["example_choice"]
    if choice == NONE_CHOICE:
        st.session_state["example_truth"] = None
        return
    row = samples.iloc[sample_labels.index(choice)]
    st.session_state["derivative"] = row["derivative"]
    for c in COLS:
        st.session_state[f"in_{c}"] = float(row[c])
    st.session_state["example_truth"] = {
        "values": {c: float(row[c]) for c in COLS},
        "codes": {k: int(row[v["column"]]) for k, v in TARGETS.items()},
    }


sample_labels = [f"{r.die_id} ({r.derivative})" for r in samples.itertuples()]
if "derivative" not in st.session_state:
    st.session_state["derivative"] = "MCU-A"
    set_defaults("MCU-A")
    st.session_state["example_choice"] = NONE_CHOICE
    st.session_state["example_truth"] = None


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
hs = cfg["fresh_data_check"]["results"]["hsi"]["within_1_%"]
hh = card["holdout_results"]["hsi"]["within_1_%"]
html(f"""
<div class="hero">
<span class="hero-badge">⚡ AI-assisted trim &amp; fuse</span>
<div class="hero-title">MCU <span>Trim-Code</span> Predictor</div>
<div class="hero-sub">Predict the best bandgap, HSI-oscillator and LDO trim codes straight from pre-trim
measurements, so the slow trial-and-error trim search can be replaced by a quick verify step.</div>
<div class="chips">
<div class="chip"><b>3</b> trim codes predicted</div>
<div class="chip"><b>99.9%</b> bandgap &amp; LDO within ±1 code</div>
<div class="chip"><b>{min(hs, hh):.0f}–{max(hs, hh):.0f}%</b> HSI within ±1 code</div>
<div class="chip">Random Forest + physics model</div>
<div class="chip">🤖 LLM project assistant</div>
</div>
</div>
""")

tab_start, tab_single, tab_chat, tab_batch, tab_fuse, tab_about = st.tabs(["📖  Start here", "🎯  Single die", "🤖  Project assistant", "📁  Batch (CSV)", "🔐  Fuse map", "ℹ️  About the model"])


# ---------------------------------------------------------------------------
# Tab 1: single die
# ---------------------------------------------------------------------------
with tab_single:
    left, right = st.columns([1, 1.2], gap="large")

    with left:
        with st.container(border=True):
            html('<div class="sec"><span class="dot"></span>1 · Enter measurements</div>')
            st.selectbox("Load an example die (unseen test data)", [NONE_CHOICE] + sample_labels,
                         key="example_choice", on_change=on_example_change,
                         help="These dies come from a freshly generated dataset the model never saw.")
            st.selectbox("Chip type", DERIVATIVES, key="derivative", on_change=on_derivative_change)

            html('<div class="mini">Measured before trimming</div>')
            c1, c2, c3 = st.columns(3)
            for col, box in zip(UNTRIM_COLS, (c1, c2, c3)):
                lab, hlp, lo, hi, step, fmt = INPUTS[col]
                with box:
                    st.number_input(lab, min_value=lo, max_value=hi, step=step, format=fmt, help=hlp, key=f"in_{col}")

            html('<div class="mini">Process-monitor readings</div>')
            p1, p2 = st.columns(2)
            for i, col in enumerate(PCM_COLS):
                lab, hlp, lo, hi, step, fmt = INPUTS[col]
                with (p1 if i % 2 == 0 else p2):
                    st.number_input(lab, min_value=lo, max_value=hi, step=step, format=fmt, help=hlp, key=f"in_{col}")

            st.button("↺  Reset to typical values for this chip type", on_click=on_derivative_change)

    # ----- prediction (live) -----
    row = {"derivative": st.session_state["derivative"], **{c: st.session_state[f"in_{c}"] for c in COLS}}
    result = predictor.predict(row).iloc[0]

    with right:
        html(f'<div class="sec"><span class="dot"></span>2 · Predicted trim codes '
             f'<span class="chip" style="margin-left:6px">{st.session_state["derivative"]}</span></div>')
        cards = st.columns(3, gap="medium")
        for key, box in zip(TARGETS, cards):
            with box:
                html(f"""
                <div class="tcard" style="--accent:{ACCENTS[key]}">
                <div class="tcard-title">{TARGET_LABELS[key]}</div>
                <div class="tcard-value">{int(result[f"{key}_trim_code"])}</div>
                <div class="tcard-sub">valid range {CODE_RANGE[key]}</div>
                <div class="tcard-note">{acc_text(key)}</div>
                </div>
                """)

        warn = result["input_warnings"]
        if warn:
            st.warning(f"Input outside the range seen in training ({warn}). The model is less reliable here.")

        # Confidence for HSI (the hard target) from the spread between trees
        lo_c, hi_c = cfg["hsi_spread_cutoffs"]
        spread = float(result["hsi_spread"])
        acc = cfg["hsi_within1_by_confidence"]
        if spread <= lo_c:
            cls, label, msg = "high", "High", (f"The forest agrees. On new test data about {acc['high']:.0f}% of such dies "
                                                "were within ±1 code. Apply the predicted code and verify once.")
        elif spread <= hi_c:
            cls, label, msg = "med", "Medium", (f"About {acc['medium']:.0f}% of such dies were within ±1 code. "
                                                 "Verify the result; it may be a code or two off.")
        else:
            cls, label, msg = "low", "Low", (f"The forest disagrees. Only about {acc['low']:.0f}% of such dies were within "
                                              "±1 code. Consider a short local search around the prediction.")
        top = hi_c * 2
        pos = min(spread / top, 1.0) * 100
        g_w, a_w = lo_c / top * 100, (hi_c - lo_c) / top * 100
        html(f"""
        <div class="conf conf-{cls}">
        <span class="pill pill-{cls}">HSI confidence: {label}</span>{msg}
        <div class="meter-wrap">
        <div class="meter">
        <div style="width:{g_w:.1f}%;background:#22c55e"></div>
        <div style="width:{a_w:.1f}%;background:#f59e0b"></div>
        <div style="flex:1;background:#ef4444"></div>
        </div>
        <div class="pin" style="left:calc({pos:.1f}% - 2px)"></div>
        </div>
        <div class="meter-labels"><span>forest agrees</span><span>forest disagrees</span></div>
        </div>
        """)
        st.caption("Bandgap and LDO are almost always within ±1 code (99.9% or better on all tests), "
                   "so a confidence flag is only shown for HSI.")

        with st.expander("🤔  Do you know what you just did? (plain-English explanation)"):
            ex = explain.explain_prediction(row, result, cfg, card)
            e0, e1, e2, e3, e4, e5 = st.tabs(["🟢 Plain words", "📝 In short", "📥 What you entered", "📤 What came out", "🛡️ How sure?", "📚 Glossary"])
            with e0:
                st.markdown(explain.simple_summary(row, result, cfg))
            with e1:
                st.markdown(ex["short"])
            with e2:
                st.markdown(ex["inputs"])
            with e3:
                st.markdown(ex["outputs"])
            with e4:
                st.markdown(ex["sure"])
            with e5:
                st.markdown(ex["glossary"])
            st.caption("Still unclear? Open the 🤖 Project assistant tab and ask. It can see this die.")

        with st.expander("🔍  How was this prediction made?"):
            phys = physics_estimate(pd.DataFrame([row])).iloc[0]
            how = pd.DataFrame({
                "Step 1: physics guess": [phys[k] for k in TARGETS],
                "Step 2: model correction": [result[f"{k}_trim_code"] - phys[k] for k in TARGETS],
                "Final code": [int(result[f"{k}_trim_code"]) for k in TARGETS],
            }, index=[TARGET_LABELS[k] for k in TARGETS]).round(2)
            st.dataframe(how)
            st.caption("The physics guess uses the step size of each code (bandgap 2 mV, HSI 0.2 %, LDO 5 mV). "
                       "A Random Forest then learns the correction from the process-monitor readings and chip type.")

        truth = st.session_state.get("example_truth")
        if truth and all(abs(truth["values"][c] - st.session_state[f"in_{c}"]) < 1e-9 for c in COLS):
            html('<div class="mini">Check against the true best codes (known for this example die)</div>')
            cmp_df = pd.DataFrame({
                "Predicted": [int(result[f"{k}_trim_code"]) for k in TARGETS],
                "True best": [truth["codes"][k] for k in TARGETS],
            }, index=[TARGET_LABELS[k] for k in TARGETS])
            cmp_df["Difference"] = cmp_df["Predicted"] - cmp_df["True best"]
            cmp_df["Within ±1?"] = cmp_df["Difference"].abs().le(1).map({True: "✅", False: "❌"})
            st.dataframe(cmp_df)


# ---------------------------------------------------------------------------
# Tab 2: batch
# ---------------------------------------------------------------------------
with tab_batch:
    with st.container(border=True):
        html('<div class="sec"><span class="dot"></span>Predict many dies from a CSV</div>')
        needed = ["derivative"] + COLS
        st.write("Your CSV needs these columns: " + ", ".join(f"`{c}`" for c in needed) + ". "
                 "Extra columns are kept in the output. If it also contains the true codes "
                 "(`bg_trim_code_opt`, `hsi_trim_code_opt`, `ldo_trim_code_opt`), accuracy is calculated.")
        template = samples[["die_id"] + needed]
        st.download_button("⬇  Download a template / example CSV", template.to_csv(index=False).encode(),
                           "mcu_trim_template.csv", "text/csv")

        up = st.file_uploader("Upload CSV", type=["csv"])
        if up is not None:
            try:
                data = pd.read_csv(up)
                preds = predictor.predict(data)
                out = pd.concat([data.reset_index(drop=True), preds], axis=1)
                st.success(f"Predicted {len(out)} dies.")
                st.dataframe(out.head(200))
                st.download_button("⬇  Download predictions (CSV)", out.to_csv(index=False).encode(),
                                   "mcu_trim_predictions.csv", "text/csv")
                truth_cols = [v["column"] for v in TARGETS.values()]
                if all(c in data.columns for c in truth_cols):
                    rows = []
                    for k, spec in TARGETS.items():
                        rows.append({"target": TARGET_LABELS[k],
                                     **evaluate_codes(data[spec["column"]], preds[f"{k}_trim_code"].to_numpy(), spec["bits"])})
                    html('<div class="mini">Accuracy on your file</div>')
                    st.dataframe(pd.DataFrame(rows).set_index("target"))
                n_warn = int((preds["input_warnings"] != "").sum())
                if n_warn:
                    st.warning(f"{n_warn} die(s) have inputs outside the training range; see the `input_warnings` column.")
            except ValueError as e:
                st.error(str(e))
            except Exception as e:  # unreadable file, etc.
                st.error(f"Could not process the file: {e}")


# ---------------------------------------------------------------------------
# Tab: fuse map (rule-based; no ML decides any fuse bit)
# ---------------------------------------------------------------------------
FUSE_NONE = "(none - type or paste my own)"
OPT_RULES = "Rules only"
OPT_PRED = "Also compare trim fields with the current die's predicted codes (±1 allowed)"
OPT_TRUE = "Also compare trim fields with the example's best codes (exact)"
fuse_samples = pd.read_csv(APP_DIR / "sample_fuse_words.csv")
fuse_labels = [f"{r.die_id} ({r.derivative})" for r in fuse_samples.itertuples()]


def on_fuse_example():
    choice = st.session_state["fuse_example"]
    if choice == FUSE_NONE:
        st.session_state["fuse_truth"] = None
        return
    r = fuse_samples.iloc[fuse_labels.index(choice)]
    st.session_state["fuse_in"] = r["fuse_map_hex"]
    st.session_state["fuse_deriv"] = r["derivative"]
    st.session_state["fuse_truth"] = {"error_type": r["error_type"],
                                      "codes": {"bg": int(r["bg_trim_code_opt"]), "hsi": int(r["hsi_trim_code_opt"]), "ldo": int(r["ldo_trim_code_opt"])}}


def show_issues(issues):
    if not issues:
        st.success("✅ All rules passed.")
        return
    for i in issues:
        box = st.error if i["severity"] == "critical" else st.warning
        box(f"**{i['field']}** ({i['type']}): {i['message']}  \n*Fix: {i['fix']}*")


with tab_fuse:
    st.session_state.setdefault("fuse_in", "")
    st.session_state.setdefault("fuse_deriv", "MCU-A")
    st.session_state.setdefault("fuse_truth", None)
    st.session_state.setdefault("ts_cal1", 1030)
    html('<div class="sec"><span class="dot"></span>Fuse map: build and check the 64-bit fuse word</div>')
    st.write("The trim codes are stored permanently in a 64-bit **fuse word**, together with chip-type settings and security bits. "
             "A wrong bit cannot be undone, so every fuse bit here is decided by **plain, auditable rules** from the fuse schema. "
             "The machine-learning model only suggests the three trim codes; it never decides a fuse bit.")
    st.caption("Synthetic data. Never burn fuses based on this tool alone: verify on the tester first.")
    fb, fc, fd = st.tabs(["🧱  Build for the current die", "🔎  Check a fuse word", "📋  Check a CSV"])

    with fb:
        with st.container(border=True):
            st.markdown(f"Chip type **{row['derivative']}** with predicted codes: bandgap **{int(result['bg_trim_code'])}**, "
                        f"HSI **{int(result['hsi_trim_code'])}**, LDO **{int(result['ldo_trim_code'])}** (from the Single die tab).")
            st.number_input("TS_CAL1: temperature-sensor reading at 30 °C (counts)", min_value=0, max_value=4095, step=1, key="ts_cal1",
                            help="A per-die measurement, not predicted by the model. In the training data it ranges from about 955 to 1104.")
            word = ft.build_fuse_word(row["derivative"], int(result["bg_trim_code"]), int(result["hsi_trim_code"]),
                                      int(result["ldo_trim_code"]), int(st.session_state["ts_cal1"]))
            html('<div class="mini">Fuse word for this die</div>')
            st.code(ft.format_word(word), language=None)
            show_issues(ft.validate(word, row["derivative"], {k: int(result[f"{k}_trim_code"]) for k in TARGETS}))
            st.dataframe(pd.DataFrame(ft.decode_table(word, row["derivative"])), hide_index=True)
            st.caption("Per-die fields (trim codes, TS_CAL1) come from the prediction and the measurement. All other fields are fixed "
                       f"by the chip type ({row['derivative']}), including DEBUG_LOCK = 1 (required in production).")

    with fc:
        with st.container(border=True):
            st.selectbox("Load an example fuse word (some contain injected errors)", [FUSE_NONE] + fuse_labels,
                         key="fuse_example", on_change=on_fuse_example)
            c1, c2 = st.columns([2, 1])
            c1.text_input("Fuse word (hex)", key="fuse_in", placeholder="0x000002FF504A2927")
            c2.selectbox("Chip type", DERIVATIVES, key="fuse_deriv")
            mode = st.radio("What to check", [OPT_RULES, OPT_PRED, OPT_TRUE], key="fuse_mode")
            truth = st.session_state["fuse_truth"]
            expected, tol = None, 0
            if mode == OPT_PRED:
                expected, tol = {k: int(result[f"{k}_trim_code"]) for k in TARGETS}, 1
            elif mode == OPT_TRUE:
                if truth:
                    expected = truth["codes"]
                else:
                    st.info("Load an example above to use its best codes.")
            if st.session_state["fuse_in"].strip():
                issues = ft.validate(st.session_state["fuse_in"], st.session_state["fuse_deriv"], expected, tol)
                show_issues(issues)
                if not any(i["type"] in ("bad_format", "unknown_derivative") for i in issues):
                    st.dataframe(pd.DataFrame(ft.decode_table(st.session_state["fuse_in"], st.session_state["fuse_deriv"])), hide_index=True)
                if truth:
                    found = [i["type"] for i in issues]
                    verdict = "caught ✅" if (truth["error_type"] in found or (truth["error_type"] == "none" and not found)) else \
                        "not detected with the current options ❌ (trim errors need a trim comparison)"
                    st.caption(f"Dataset label for this example: **{truth['error_type']}** → {verdict}")
            else:
                st.info("Paste a fuse word or load an example.")

    with fd:
        with st.container(border=True):
            st.write("Upload a CSV with columns `derivative` and `fuse_map_hex`. Optional: `die_id`, the best codes "
                     "(`bg_trim_code_opt`, `hsi_trim_code_opt`, `ldo_trim_code_opt`) to also check the trim fields, and `has_error` to measure detection.")
            st.download_button("⬇  Download example CSV", fuse_samples.to_csv(index=False).encode(), "sample_fuse_words.csv", "text/csv")
            upf = st.file_uploader("Upload fuse words (CSV)", type=["csv"], key="fuse_csv")
            if upf is not None:
                try:
                    fdf = pd.read_csv(upf)
                    need = [c for c in ("derivative", "fuse_map_hex") if c not in fdf.columns]
                    if need:
                        raise ValueError(f"Missing columns: {need}")
                    code_cols = ["bg_trim_code_opt", "hsi_trim_code_opt", "ldo_trim_code_opt"]
                    have_codes = all(c in fdf.columns for c in code_cols)
                    found_types, details = [], []
                    for _, r in fdf.iterrows():
                        exp_c = {"bg": r[code_cols[0]], "hsi": r[code_cols[1]], "ldo": r[code_cols[2]]} if have_codes else None
                        iss = ft.validate(r["fuse_map_hex"], r["derivative"], exp_c)
                        found_types.append(", ".join(i["type"] for i in iss))
                        details.append(" | ".join(i["message"] for i in iss))
                    out = fdf.copy()
                    out["status"] = ["OK" if not t else "FLAGGED" for t in found_types]
                    out["issues"], out["details"] = found_types, details
                    n_flag = int((out["status"] == "FLAGGED").sum())
                    st.success(f"Checked {len(out)} fuse words: {n_flag} flagged, {len(out) - n_flag} passed.")
                    if not have_codes:
                        st.caption("No best-code columns found, so corrupted trim fields cannot be detected (rules only).")
                    if "has_error" in out.columns:
                        truth_err = out["has_error"].astype(str).str.lower().isin(["true", "1"])
                        flagged = out["status"] == "FLAGGED"
                        st.markdown(f"**Detection:** caught **{int((flagged & truth_err).sum())} of {int(truth_err.sum())}** labelled errors, "
                                    f"**{int((flagged & ~truth_err).sum())}** false alarms.")
                    st.dataframe(out.head(200), hide_index=True)
                    st.download_button("⬇  Download report (CSV)", out.to_csv(index=False).encode(), "fuse_check_report.csv", "text/csv")
                except Exception as e:  # noqa: BLE001
                    st.error(f"Could not process the file: {e}")


# ---------------------------------------------------------------------------
# Tab 3: about
# ---------------------------------------------------------------------------
with tab_about:
    with st.container(border=True):
        html('<div class="sec"><span class="dot"></span>What this model does</div>')
        st.write("Chips come out of the factory slightly off-target. Each one is normally trimmed by a trial-and-error "
                 "(binary) search, about 20 test steps per chip. This model predicts the best code directly from "
                 "measurements taken before trimming, so only a short verify step is needed.")

        html('<div class="sec"><span class="dot"></span>How it works</div>')
        html("""
        <div class="steps">
        <div class="step"><div class="n">1</div><br><b>Physics guess</b>
        <div class="t">Arithmetic from each code's step size gives a starting code.</div></div>
        <div class="step"><div class="n">2</div><br><b>Random Forest correction</b>
        <div class="t">One forest per target learns the leftover correction from the untrimmed measurements,
        five process monitors and the chip type.</div></div>
        <div class="step"><div class="n">3</div><br><b>Round, clip, grade</b>
        <div class="t">The result is rounded to a valid code and a confidence signal is added from how much the trees disagree.</div></div>
        </div>
        """)

    with st.container(border=True):
        html('<div class="sec"><span class="dot"></span>How accurate is it? (dies within ±1 code of the best)</div>')
        hold = card["holdout_results"]
        fresh = cfg["fresh_data_check"]["results"]
        table = pd.DataFrame({
            "Held-out wafers (same lots as training)": [hold[k]["within_1_%"] for k in TARGETS],
            "Brand-new dataset (new lots)": [fresh[k]["within_1_%"] for k in TARGETS],
            "Exact match on new dataset": [fresh[k]["exact_%"] for k in TARGETS],
        }, index=[TARGET_LABELS[k] for k in TARGETS])
        st.dataframe(table.style.format("{:.1f}%"))
        st.bar_chart(table[["Held-out wafers (same lots as training)", "Brand-new dataset (new lots)"]])
        st.caption("Bandgap and LDO are essentially solved. HSI is the hard one: the leftover error is measurement noise "
                   "that no input explains, and accuracy drops on lots the model has not seen.")

        html('<div class="sec"><span class="dot"></span>Does it transfer to a new chip type? (MCU-C treated as new)</div>')
        tl = pd.read_csv(APP_DIR / "models" / "transfer_summary.csv")
        plan = [("Physics formula only", "physics formula only", 0),
                ("Trained on MCU-A + B only (no MCU-C dies)", "A+B only (zero-shot)", 0),
                ("Plus 50 MCU-C dies", "A+B + k MCU-C dies (transfer)", 50),
                ("Only 50 MCU-C dies (from scratch)", "k MCU-C dies only (from scratch)", 50),
                ("Plus all 421 MCU-C dies", "A+B + k MCU-C dies (transfer)", 421)]
        tr_rows = {}
        for label, setup, k in plan:
            sub = tl[(tl["setup"] == setup) & (tl["k"] == k)].set_index("key")["within_1_%"]
            tr_rows[label] = {"Bandgap": sub["bg"], "HSI": sub["hsi"], "LDO": sub["ldo"]}
        st.dataframe(pd.DataFrame(tr_rows).T.style.format("{:.1f}%"))
        st.caption("Share of MCU-C dies within ±1 code. The model already works on the new chip type without any of its data "
                   "(HSI 78.5% vs 79.1% with a full wafer), and a few extra dies did not help. Building from scratch on a few dies is clearly worse. "
                   "Chip-type-specific tuning should matter for package-dependent drift, which is future work.")

        html('<div class="sec"><span class="dot"></span>Limitations</div>')
        for item in card["limitations"]:
            st.markdown(f"- {item}")
        st.caption(f"Model {card['model_name']} v{card['version']}, trained {card['trained_on']} on "
                   f"{card['training_rows']} dies / {card['training_wafers']} wafers.")


# ---------------------------------------------------------------------------
# Tab 2: project assistant (LLM mode + offline fallback)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_assistant():
    kb = assistant.load_kb(APP_DIR / "project_knowledge.md")
    simple = assistant.load_kb(APP_DIR / "simple_guide.md")
    return kb, assistant.OfflineAnswerer(kb), simple, assistant.OfflineAnswerer(simple)


SUGGESTED_DETAILED = ["Explain my current prediction in simple words",
             "Why did you choose Random Forest?",
             "What is trimming and why does it matter?",
             "What are the limitations of this model?",
             "Is the model offline or online?",
             "How does this assistant work?"]
SUGGESTED_SIMPLE = ["What is this project about, in the simplest words?",
                    "Explain my current prediction in simple words",
                    "What is trimming?",
                    "What is a Random Forest?",
                    "What is a fuse?",
                    "How much can I trust the predictions?"]


def queue_question(q):
    st.session_state["pending_q"] = q


def clear_chat():
    st.session_state["chat"] = []


with tab_chat:
    kb_text, offline, simple_text, offline_simple = load_assistant()
    st.session_state.setdefault("chat", [])
    st.session_state.setdefault("llm_count", 0)
    try:
        secrets = st.secrets
    except Exception:  # no secrets file is normal
        secrets = None
    api_key = assistant.get_api_key(st.session_state.get("user_api_key"), secrets)
    model = assistant.get_model(secrets)
    sdk_ok = assistant.llm_available()
    within_limit = st.session_state["llm_count"] < assistant.MAX_LLM_QUESTIONS
    llm_ready = bool(api_key) and sdk_ok and within_limit

    badge = ('<span class="mode mode-llm">● LLM mode</span>' if llm_ready
             else '<span class="mode mode-off">● Offline mode</span>')
    html(f'<div class="sec"><span class="dot"></span>Project assistant {badge}</div>')
    if llm_ready:
        st.caption(f"Answers are generated by a language model ({model}) from the project notes and the die on the "
                   f"Single die tab. LLM questions used this session: {st.session_state['llm_count']}/{assistant.MAX_LLM_QUESTIONS}.")
    elif api_key and not sdk_ok:
        st.caption("An API key was found but the `anthropic` package is not installed (`pip install anthropic`). Using offline mode.")
    elif api_key and not within_limit:
        st.caption("Session limit for LLM questions reached. Using offline mode for the rest of this session.")
    else:
        st.caption("Offline mode: answers come straight from the project notes. Connect an API key below for free-form, "
                   "conversational answers.")

    level = st.radio("How should answers be explained?", ["🟢 Simple (anyone can follow)", "🔵 Detailed (technical)"],
                     key="chat_level", horizontal=True)
    is_simple = level.startswith("🟢")
    active_kb = simple_text if is_simple else kb_text
    active_off = offline_simple if is_simple else offline
    SUGGESTED = SUGGESTED_SIMPLE if is_simple else SUGGESTED_DETAILED

    with st.expander("🔑  Connect an LLM (optional)"):
        st.text_input("Anthropic API key", type="password", key="user_api_key",
                      help="Kept only in this browser session. Never written to a file.")
        st.markdown("Or set it before starting the app: `ANTHROPIC_API_KEY` (environment variable) or "
                    "`.streamlit/secrets.toml`. Never put a key in the code or on GitHub.")

    st.markdown("**Try one of these:**")
    for r0 in range(0, len(SUGGESTED), 3):
        cols = st.columns(3)
        for j, q in enumerate(SUGGESTED[r0:r0 + 3]):
            cols[j].button(q, on_click=queue_question, args=(q,), key=f"sug_{r0 + j}")

    for m in st.session_state["chat"]:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    pending = st.session_state.pop("pending_q", None)
    typed = st.chat_input("Ask about the project, the data, the model or your prediction...")
    question = typed or pending
    if question:
        question = question[:assistant.MAX_USER_CHARS]
        st.session_state["chat"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        ex_now = explain.explain_prediction(row, result, cfg, card)
        cur_summary = explain.simple_summary(row, result, cfg) if is_simple else ex_now["short"] + "\n\n" + ex_now["outputs"]
        offline_fn = lambda: active_off.answer(question, current_summary=cur_summary)
        with st.chat_message("assistant"):
            if llm_ready:
                st.session_state["llm_count"] += 1
                system = assistant.build_system_prompt(active_kb, explain.current_context_text(row, result, cfg), "simple" if is_simple else "detailed")
                answer = st.write_stream(assistant.stream_llm_answer(
                    api_key, model, system, st.session_state["chat"], offline_fallback=offline_fn))
            else:
                answer = offline_fn()
                st.markdown(answer)
        st.session_state["chat"].append({"role": "assistant", "content": answer})

    if st.session_state["chat"]:
        st.button("🧹  Clear chat", on_click=clear_chat)
    st.caption("The assistant explains the project. It cannot change the model or its predictions. "
               "Double-check any number you plan to quote against the notebooks.")


# ---------------------------------------------------------------------------
# Tab 0: start here (plain-language tour for anyone)
# ---------------------------------------------------------------------------
with tab_start:
    _kb, _off, simple_all, _off_s = load_assistant()
    guide = dict(assistant.split_chunks(simple_all))
    html('<div class="sec"><span class="dot"></span>Start here: what is this app, in plain words?</div>')
    st.write("Factories make millions of tiny computer chips. Like cookies from the same dough, every chip comes out slightly different, "
             "so each one needs a small adjustment, a bit like tuning a guitar string. Finding that adjustment by trial and error takes about "
             "20 tries per chip. This app shows a program that looks at a chip's measurements and tells you the adjustment straight away.")
    html("""
    <div class="steps">
    <div class="step"><div class="n">1</div><br><b>A chip is made</b>
    <div class="t">Like cookies on a tray, each chip comes out a tiny bit different from the others.</div></div>
    <div class="step"><div class="n">2</div><br><b>We measure it</b>
    <div class="t">Three circuits are measured: a steady voltage ruler, a built-in clock and a power regulator. A few "birth certificate" readings say how it was made.</div></div>
    <div class="step"><div class="n">3</div><br><b>The program predicts the clicks</b>
    <div class="t">Each circuit has an adjustment dial. A model that studied 8,420 chips says how many clicks each dial needs.</div></div>
    <div class="step"><div class="n">4</div><br><b>A tester checks once</b>
    <div class="t">Instead of about 20 trial-and-error steps, the tester only verifies the answer, about 5 checks in our tests.</div></div>
    <div class="step"><div class="n">5</div><br><b>The setting is made permanent</b>
    <div class="t">The dial positions are written into tiny one-time switches called fuses, like writing in ink instead of pencil.</div></div>
    <div class="step"><div class="n">6</div><br><b>A rulebook double-checks</b>
    <div class="t">A fuse checker reads the permanent settings and warns in plain words about any mistake before it is too late.</div></div>
    </div>
    """)
    with st.container(border=True):
        html('<div class="sec"><span class="dot"></span>Words in plain English</div>')
        st.markdown(
            "| Word | What it means in everyday life |\n|---|---|\n"
            "| Chip / die | One tiny electronic brain, the size of a fingernail or smaller |\n"
            "| Wafer / lot | The big disc many chips are made on (like a pizza base) / a batch of discs (like several trays from one oven session) |\n"
            "| Trimming | The final fine adjustment of a chip, like tuning a guitar |\n"
            "| Trim code | The click position of an adjustment dial |\n"
            "| Bandgap | The chip's super-steady voltage ruler |\n"
            "| HSI oscillator | The chip's built-in clock, like a metronome |\n"
            "| LDO | A power regulator, like a pressure regulator on a water tap |\n"
            "| Fuse | A one-time switch that makes a setting permanent |\n"
            "| Model | The program that learned from examples to predict the clicks |\n"
            "| Random Forest | 300 simple advisors voting on the answer |\n"
            "| Within ±1 code | At most one click away from the best answer |\n")
    with st.container(border=True):
        html('<div class="sec"><span class="dot"></span>Try it in three clicks</div>')
        st.markdown("1. Open the **🎯 Single die** tab and pick an example chip from the first dropdown.\n"
                    "2. Look at the three big numbers: they are the dial positions the program predicts.\n"
                    "3. Open the **🤔 Do you know what you just did?** drop-down and read the **🟢 Plain words** tab.\n\n"
                    "Stuck on any word? Open **🤖 Project assistant**: it answers in simple language by default.")
    hold_hsi = card["holdout_results"]["hsi"]["within_1_%"]
    fresh_hsi = cfg["fresh_data_check"]["results"]["hsi"]["within_1_%"]
    st.info(f"**How much can you trust it?** For the voltage ruler and the power regulator the prediction is within one click of the best answer "
            f"for about 99.9% of chips. For the clock it is about {min(hold_hsi, fresh_hsi):.0f} to {max(hold_hsi, fresh_hsi):.0f}%. "
            "Everything here uses practice data made by a computer program (not real chips), and a tester must always verify before anything is made permanent.")
    for title, key in [("🧒  Explain it like I'm ten years old", "Explain it like I am ten years old"),
                       ("👵  Explain it like I'm talking to my grandmother", "Explain it like I am talking to my grandmother"),
                       ("⚠️  What are the limits?", "What are the real-world limits of this project?")]:
        with st.expander(title):
            st.markdown(guide[key])
