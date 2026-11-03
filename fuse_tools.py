"""
fuse_tools.py
=============
Build, decode and validate the 64-bit MCU fuse word (the "fuse" half of Trim & Fuse).

IMPORTANT DESIGN CHOICE: everything here is plain, deterministic rules taken from the fuse schema
(mcu_fuse_schema.csv). No machine learning decides any fuse bit. Only the three trim codes come from the
ML model, and even those are checked here before a fuse word is produced. Security fuses (debug lock,
read-out protection) must always be on rule-based, auditable checks.

Bit layout (LSB first):
  0-5   BG_TRIM      bandgap trim code (6 bit)
  6-12  HSI_TRIM     HSI oscillator trim code (7 bit)
  13-17 LDO_TRIM     LDO trim code (5 bit)
  18-29 TS_CAL1      temperature-sensor reading at 30 C (12 bit, per die)
  30-31 FLASH_SIZE   0=256 KB, 1=512 KB, 2=1 MB
  32-39 PERIPH_EN    peripheral enable mask
  40    TEMP_GRADE   1 = automotive
  41    DEBUG_LOCK   1 = debug port locked (required in production)
  42    RDP          1 = read-out protection enabled
  43-63 RESERVED     must be 0
"""
import re

FIELDS = [  # name, lsb, nbits, description
    ("BG_TRIM", 0, 6, "Bandgap trim code"),
    ("HSI_TRIM", 6, 7, "HSI oscillator trim code"),
    ("LDO_TRIM", 13, 5, "LDO trim code"),
    ("TS_CAL1", 18, 12, "Temperature-sensor reading at 30 C (per die)"),
    ("FLASH_SIZE", 30, 2, "Flash size (0=256 KB, 1=512 KB, 2=1 MB)"),
    ("PERIPH_EN", 32, 8, "Peripheral enable mask"),
    ("TEMP_GRADE", 40, 1, "Temperature grade (1 = automotive)"),
    ("DEBUG_LOCK", 41, 1, "Debug port locked (required in production)"),
    ("RDP", 42, 1, "Read-out protection enabled"),
    ("RESERVED", 43, 21, "Reserved, must be 0"),
]
FIELD = {n: (lsb, nb, d) for n, lsb, nb, d in FIELDS}
FLASH_TEXT = {0: "256 KB", 1: "512 KB", 2: "1 MB"}

# What each chip type must have (from the schema). Trim codes and TS_CAL1 are per die, so not listed.
EXPECTED = {
    "MCU-A": {"FLASH_SIZE": 1, "PERIPH_EN": 0xFF, "TEMP_GRADE": 0, "DEBUG_LOCK": 1, "RDP": 0, "RESERVED": 0},
    "MCU-B": {"FLASH_SIZE": 0, "PERIPH_EN": 0x3F, "TEMP_GRADE": 0, "DEBUG_LOCK": 1, "RDP": 0, "RESERVED": 0},
    "MCU-C": {"FLASH_SIZE": 2, "PERIPH_EN": 0xFF, "TEMP_GRADE": 1, "DEBUG_LOCK": 1, "RDP": 1, "RESERVED": 0},
}
TRIM_FIELDS = {"BG_TRIM": "bg", "HSI_TRIM": "hsi", "LDO_TRIM": "ldo"}
HEX_RE = re.compile(r"^0[xX][0-9a-fA-F]{1,16}$")


def parse_word(word):
    """Accept '0x...' text (or an int) and return an int; raise ValueError with a clear message."""
    if isinstance(word, int):
        value = word
    else:
        text = str(word).strip()
        if not HEX_RE.match(text):
            raise ValueError("Fuse word must look like 0x followed by up to 16 hex digits, e.g. 0x000002FF504A2927")
        value = int(text, 16)
    if not 0 <= value < 2 ** 64:
        raise ValueError("Fuse word must fit in 64 bits.")
    return value


def decode(word):
    """Split a fuse word into its named fields (dict of ints)."""
    v = parse_word(word)
    return {n: (v >> lsb) & ((1 << nb) - 1) for n, lsb, nb, _ in FIELDS}


def build_fuse_word(derivative, bg, hsi, ldo, ts_cal1):
    """Assemble the fuse word for a die: per-die values (trim codes, TS_CAL1) + the fixed settings of its chip type."""
    if derivative not in EXPECTED:
        raise ValueError(f"Unknown derivative {derivative!r}; allowed: {sorted(EXPECTED)}")
    values = dict(EXPECTED[derivative])
    for name, v in (("BG_TRIM", bg), ("HSI_TRIM", hsi), ("LDO_TRIM", ldo), ("TS_CAL1", ts_cal1)):
        nb = FIELD[name][1]
        if int(v) != v or not 0 <= v < 2 ** nb:
            raise ValueError(f"{name} must be a whole number from 0 to {2 ** nb - 1} (got {v}).")
        values[name] = int(v)
    word = 0
    for name, lsb, nb, _ in FIELDS:
        word |= (values[name] & ((1 << nb) - 1)) << lsb
    return word


def format_word(word):
    return f"0x{parse_word(word):016X}"


def validate(word, derivative, expected_codes=None, trim_tolerance=0):
    """Check a fuse word against the rules of its chip type.

    expected_codes: optional {'bg': int, 'hsi': int, 'ldo': int} (the best codes from the tester or the model).
                    If given, the trim fields are compared with them (difference > trim_tolerance is flagged).
    Returns a list of issues; an empty list means every rule passed.
    Each issue: {type, field, severity, message, fix}. 'type' uses the same names as the dataset's audit file.
    """
    issues = []
    if derivative not in EXPECTED:
        return [{"type": "unknown_derivative", "field": "-", "severity": "error",
                 "message": f"Unknown chip type {derivative!r}.", "fix": f"Use one of {sorted(EXPECTED)}."}]
    try:
        f = decode(word)
    except ValueError as e:
        return [{"type": "bad_format", "field": "-", "severity": "error", "message": str(e), "fix": "Correct the fuse word."}]
    exp = EXPECTED[derivative]

    if f["FLASH_SIZE"] != exp["FLASH_SIZE"]:
        issues.append({"type": "wrong_flash_size", "field": "FLASH_SIZE", "severity": "error",
                       "message": f"FLASH_SIZE is {f['FLASH_SIZE']} ({FLASH_TEXT.get(f['FLASH_SIZE'], 'invalid')}) but {derivative} must be "
                                  f"{exp['FLASH_SIZE']} ({FLASH_TEXT[exp['FLASH_SIZE']]}).",
                       "fix": f"Set bits 30-31 to {exp['FLASH_SIZE']}."})
    if f["PERIPH_EN"] != exp["PERIPH_EN"]:
        issues.append({"type": "periph_mask_mismatch", "field": "PERIPH_EN", "severity": "error",
                       "message": f"Peripheral enable mask is 0x{f['PERIPH_EN']:02X} but {derivative} must be 0x{exp['PERIPH_EN']:02X}. "
                                  "Wrong peripherals would be enabled or disabled.",
                       "fix": f"Set bits 32-39 to 0x{exp['PERIPH_EN']:02X}."})
    if f["TEMP_GRADE"] != exp["TEMP_GRADE"]:
        issues.append({"type": "temp_grade_mismatch", "field": "TEMP_GRADE", "severity": "error",
                       "message": f"TEMP_GRADE is {f['TEMP_GRADE']} but {derivative} must be {exp['TEMP_GRADE']} "
                                  f"({'automotive' if exp['TEMP_GRADE'] else 'not automotive'}).",
                       "fix": f"Set bit 40 to {exp['TEMP_GRADE']}."})
    if f["DEBUG_LOCK"] != 1:
        issues.append({"type": "missing_debug_lock", "field": "DEBUG_LOCK", "severity": "critical",
                       "message": "DEBUG_LOCK is 0: the debug port would stay open on a production part (security risk; the part could be read out or tampered with).",
                       "fix": "Set bit 41 to 1."})
    if f["RDP"] != exp["RDP"]:
        if exp["RDP"] == 1:
            issues.append({"type": "rdp_missing_automotive", "field": "RDP", "severity": "critical",
                           "message": f"Read-out protection (RDP) is off but {derivative} is an automotive part and must have it on (security risk).",
                           "fix": "Set bit 42 to 1."})
        else:
            issues.append({"type": "rdp_unexpected", "field": "RDP", "severity": "error",
                           "message": f"RDP is on but {derivative} should have it off. The part could not be re-read or re-tested as intended.",
                           "fix": "Set bit 42 to 0."})
    if f["RESERVED"] != 0:
        issues.append({"type": "reserved_bit_set", "field": "RESERVED", "severity": "error",
                       "message": f"Reserved bits (43-63) must be 0 but hold 0x{f['RESERVED']:X}. Reserved bits may be used by future silicon.",
                       "fix": "Clear bits 43-63."})

    if expected_codes:
        bad = []
        for fld, key in TRIM_FIELDS.items():
            if key in expected_codes and expected_codes[key] is not None:
                diff = f[fld] - int(expected_codes[key])
                if abs(diff) > trim_tolerance:
                    bad.append(f"{fld} is {f[fld]} but expected {int(expected_codes[key])} (difference {diff:+d})")
        if bad:
            issues.append({"type": "trim_field_corrupt", "field": "BG/HSI/LDO_TRIM", "severity": "error",
                           "message": "Trim field(s) differ from the expected codes: " + "; ".join(bad) + ".",
                           "fix": "Re-check the codes on the tester and rewrite the trim fields."})
    return issues


def decode_table(word, derivative=None):
    """Rows for display: field, bits, value, expected (if known)."""
    f = decode(word)
    rows = []
    for name, lsb, nb, desc in FIELDS:
        exp = EXPECTED.get(derivative, {}).get(name) if derivative else None
        hi = lsb + nb - 1
        rows.append({"Field": name, "Bits": f"{lsb}" if nb == 1 else f"{lsb}-{hi}", "Value": str(f[name]),
                     "Expected": "per die" if exp is None else str(exp), "Meaning": desc})
    return rows
