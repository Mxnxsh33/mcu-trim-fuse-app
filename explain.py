"""
explain.py
==========
Turns a prediction into plain English ("Do you know what you just did?").

Everything here is computed from the actual inputs/outputs of the current die, using the
nominal step size of each trim code (from the fuse schema): bandgap 2 mV per step (centre 32),
HSI oscillator 0.2 % per step (centre 64), LDO 5 mV per step (centre 16). On the training data these
nominal steps reproduce the real post-trim values to within about one step, so the
"expected result after trim" numbers are honest estimates, not guarantees.
"""

TARGET = {"bg": 1.200, "hsi": 16.0, "ldo": 1.100}   # what each circuit should end up at
CENTRE = {"bg": 32, "hsi": 64, "ldo": 16}            # code that means "no adjustment"
LABEL = {"bg": "Bandgap reference", "hsi": "HSI oscillator (internal clock)", "ldo": "LDO regulator"}


# ----------------------------------------------------------------------------
# small helpers
# ----------------------------------------------------------------------------
def _gap_text(key, value):
    """How far the measured value is from its target, in words."""
    if key == "bg":
        d = (value - TARGET["bg"]) * 1000
        pct = d / (TARGET["bg"] * 1000) * 100
        if abs(d) < 0.5:
            return "already almost exactly on target"
        return f"{abs(d):.1f} mV too {'high' if d > 0 else 'low'} ({abs(pct):.2f}% {'above' if d > 0 else 'below'} target)"
    if key == "hsi":
        pct = (value - TARGET["hsi"]) / TARGET["hsi"] * 100
        if abs(pct) < 0.1:
            return "already almost exactly on target"
        return f"{abs(pct):.1f}% too {'fast' if pct > 0 else 'slow'}"
    d = (value - TARGET["ldo"]) * 1000
    if abs(d) < 0.5:
        return "already almost exactly on target"
    return f"{abs(d):.1f} mV too {'high' if d > 0 else 'low'}"


def _expected_after(key, value, code):
    steps = code - CENTRE[key]
    if key == "bg":
        return value + steps * 0.002
    if key == "hsi":
        return value * (1 + 0.002 * steps)
    return value + steps * 0.005


def _fmt_val(key, v):
    return f"{v:.4f} V" if key == "bg" else (f"{v:.3f} MHz" if key == "hsi" else f"{v:.4f} V")


def _what_code_does(key, code):
    steps = code - CENTRE[key]
    if steps == 0:
        return f"Centre code ({CENTRE[key]}): no adjustment"
    way = "up" if steps > 0 else "down"
    n = abs(steps)
    if key == "bg":
        return f"{n} step{'s' if n != 1 else ''} {way} from centre ({CENTRE[key]}); each step is about 2 mV, so it moves the voltage {way} by about {n * 2} mV"
    if key == "hsi":
        return f"{n} step{'s' if n != 1 else ''} {way} from centre ({CENTRE[key]}); each step is about 0.2%, so it moves the clock {way} by about {n * 0.2:.1f}%"
    return f"{n} step{'s' if n != 1 else ''} {way} from centre ({CENTRE[key]}); each step is about 5 mV, so it moves the voltage {way} by about {n * 5} mV"


def confidence_level(spread, cfg):
    """(name, css class, percent within +-1 seen on new data) from the tree-spread of the HSI forest."""
    lo, hi = cfg["hsi_spread_cutoffs"]
    acc = cfg["hsi_within1_by_confidence"]
    if spread <= lo:
        return "High", "high", acc["high"]
    if spread <= hi:
        return "Medium", "med", acc["medium"]
    return "Low", "low", acc["low"]


def current_context_text(row, result, cfg):
    """Compact plain-text description of the current die, handed to the chat assistant."""
    name, _, pct = confidence_level(float(result["hsi_spread"]), cfg)
    lines = [f"Chip type: {row['derivative']}",
             f"Bandgap before trim: {row['bg_untrim_V']:.5f} V (target 1.200 V; {_gap_text('bg', row['bg_untrim_V'])})",
             f"HSI oscillator before trim: {row['hsi_untrim_MHz']:.4f} MHz (target 16 MHz; {_gap_text('hsi', row['hsi_untrim_MHz'])})",
             f"LDO before trim: {row['ldo_untrim_V']:.5f} V (target 1.100 V; {_gap_text('ldo', row['ldo_untrim_V'])})",
             "Process monitors: " + ", ".join(f"{k}={row[k]}" for k in
                                              ["pcm_vth_n_mV", "pcm_vth_p_mV", "pcm_rs_ohm_sq", "pcm_tox_A", "pcm_mobility_rel"]),
             f"Predicted trim codes: bandgap={int(result['bg_trim_code'])}, HSI={int(result['hsi_trim_code'])}, LDO={int(result['ldo_trim_code'])}",
             f"HSI confidence: {name} (about {pct:.0f}% of such dies were within +-1 code on new test data)"]
    if result["input_warnings"]:
        lines.append(f"Input warnings: {result['input_warnings']}")
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# the explanation itself (returned as one markdown string per tab)
# ----------------------------------------------------------------------------
def explain_prediction(row, result, cfg, card):
    codes = {k: int(result[f"{k}_trim_code"]) for k in TARGET}
    vals = {"bg": row["bg_untrim_V"], "hsi": row["hsi_untrim_MHz"], "ldo": row["ldo_untrim_V"]}
    old_checks = 19.8
    new_checks = sum(card["holdout_results"][k]["ai_iterations"] for k in TARGET)

    short = (
        f"**In one sentence:** you gave the system measurements from one **{row['derivative']}** microcontroller chip, "
        "taken *before* its factory adjustment (called **trimming**). It predicted the three adjustment settings "
        "(**trim codes**) that should bring the chip's reference voltage, internal clock and voltage regulator onto their targets.\n\n"
        f"**Why this is useful:** finding the right settings normally takes a trial-and-error search of about "
        f"{old_checks:.0f} test steps per chip. A good prediction lets the factory try one educated guess and verify it. "
        f"On held-out test chips the average dropped from about {old_checks:.0f} to about {new_checks:.0f} checks per chip "
        "(using the dataset's own counting rule).\n\n"
        "**What you are actually finding:** for each of the three circuits, *the one whole number (code) that moves the circuit "
        "closest to its target.* The codes below are the answer.\n\n"
        f"**Your answer:** bandgap code **{codes['bg']}**, HSI oscillator code **{codes['hsi']}**, LDO code **{codes['ldo']}**."
    )

    in_rows = "".join(
        f"| {LABEL[k]} before trim | {_fmt_val(k, vals[k])} | {_fmt_val(k, TARGET[k])} | {_gap_text(k, vals[k])} |\n"
        for k in TARGET)
    pcm_info = [
        ("pcm_vth_n_mV", "NMOS threshold voltage", "mV", "how easily the chip's N-type transistors switch on"),
        ("pcm_vth_p_mV", "PMOS threshold voltage", "mV", "how easily the P-type transistors switch on (negative by convention)"),
        ("pcm_rs_ohm_sq", "Sheet resistance", "ohm/sq", "how strongly the thin conducting layers resist current"),
        ("pcm_tox_A", "Oxide thickness", "angstrom", "thickness of the thin insulating layer (an angstrom is one ten-billionth of a metre). The HSI prediction relies on this one a lot"),
        ("pcm_mobility_rel", "Relative carrier mobility", "", "how easily charge moves through the silicon; 1.0 means nominal"),
    ]
    pcm_rows = "".join(f"| {n} | {row[c]} {u} | {m} |\n" for c, n, u, m in pcm_info)
    inputs = (
        f"**Chip type:** {row['derivative']} (one of three versions of the microcontroller; the model treats them slightly differently).\n\n"
        "**1. Measured before trimming** (how far off the chip is right now)\n\n"
        "| Circuit | Your value | Target | What it tells us |\n|---|---|---|---|\n" + in_rows +
        "\n**2. Process-monitor readings** (the chip's \"birth certificate\": measurements of how this part of the wafer was manufactured, "
        "taken on tiny test structures next to the chips)\n\n"
        "| Reading | Your value | What it means |\n|---|---|---|\n" + pcm_rows +
        "\nThe idea: chips made under similar factory conditions need similar adjustments, so these readings help the model correct its guess, "
        "especially for the HSI clock."
    )

    out_rows = ""
    for k in TARGET:
        exp = _expected_after(k, vals[k], codes[k])
        out_rows += (f"| {LABEL[k]} | **{codes[k]}** | {_what_code_does(k, codes[k])} | "
                     f"about {_fmt_val(k, exp)} (target {_fmt_val(k, TARGET[k])}) |\n")
    outputs = (
        "**A trim code is a whole number that is stored permanently in the chip (in tiny fuses) and nudges one circuit up or down.** "
        "Each circuit has a \"centre\" code that means *no adjustment*; going above or below it adjusts the circuit in small, fixed steps.\n\n"
        "| Circuit | Predicted code | What the code does | Expected result after trimming |\n|---|---|---|---|\n" + out_rows +
        "\n*The expected values use the nominal step size, so a real chip can differ by about one step. That is why the model is judged "
        "on being within ±1 code of the best code.*\n\n"
        "**How the answer was produced:** (1) arithmetic from the step sizes gives a first guess, (2) a Random Forest model, trained on "
        "8,420 example chips, adds a small correction using the process-monitor readings and chip type.\n\n"
        "**What happens next:** the three codes are packed into the chip's 64-bit fuse word. Open the 🔐 Fuse map tab to see that word and the rules that check it."
    )

    name, _, pct = confidence_level(float(result["hsi_spread"]), cfg)
    fresh = cfg["fresh_data_check"]["results"]
    hold = card["holdout_results"]
    sure = (
        f"**Bandgap and LDO:** very reliable. The predicted code was within ±1 of the best code for about "
        f"{min(hold['bg']['within_1_%'], fresh['bg']['within_1_%']):.1f}% (bandgap) and {min(hold['ldo']['within_1_%'], fresh['ldo']['within_1_%']):.0f}% (LDO) "
        "of test chips. Being one code off only misses the target by one small step.\n\n"
        f"**HSI clock:** harder, because its measurement is noisy. Overall it was within ±1 code for about "
        f"{min(hold['hsi']['within_1_%'], fresh['hsi']['within_1_%']):.0f}–{max(hold['hsi']['within_1_%'], fresh['hsi']['within_1_%']):.0f}% of test chips "
        "(lower on lots the model had never seen).\n\n"
        f"**For this die the HSI confidence is {name}.** The model is really 300 decision trees voting; when they agree the answer is more "
        f"trustworthy. On new test data, about {pct:.0f}% of dies with this level of agreement were within ±1 code.\n\n"
        "**Important:** this is a decision-support tool. Fuses are permanent, so the predicted codes should be verified on the tester "
        "before anything is burned into a chip. The data is synthetic."
    )
    if result["input_warnings"]:
        sure += f"\n\n**Warning for this die:** {result['input_warnings']}, so the model is extrapolating and may be less reliable."

    glossary = (
        "- **Die / chip:** one individual microcontroller cut from a silicon wafer.\n"
        "- **Wafer / lot:** a wafer is a silicon disc with hundreds of dies; a lot is a batch of wafers made together.\n"
        "- **Trimming:** fine-tuning a chip after manufacturing so its circuits hit their targets.\n"
        "- **Trim code:** the whole number that sets that fine-tuning. Higher or lower moves the circuit in fixed steps.\n"
        "- **Fuse:** a tiny one-time-programmable memory cell. Trim codes are stored in fuses, so they cannot be undone.\n"
        "- **Bandgap reference:** a circuit that makes a very stable voltage (target 1.200 V) that other circuits measure against.\n"
        "- **HSI oscillator:** the chip's internal high-speed clock (target 16 MHz).\n"
        "- **LDO:** low-dropout voltage regulator, which supplies a steady voltage (target 1.100 V) to the chip's core.\n"
        "- **Process monitor (PCM):** test structures that record how the factory process behaved for that part of the wafer.\n"
        "- **Chip type (MCU-A/B/C):** three versions of the microcontroller in the dataset.\n"
        "- **Within ±1 code:** the prediction is at most one step away from the best code. This is the main accuracy measure."
    )
    return {"short": short, "inputs": inputs, "outputs": outputs, "sure": sure, "glossary": glossary}
