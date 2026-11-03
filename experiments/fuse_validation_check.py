"""
Reproduces the fuse-validator numbers.
Usage:  python experiments/fuse_validation_check.py <folder with mcu_fuse_map_audit.csv and mcu_trim_fuse.csv>
Run from the project root so that fuse_tools.py can be imported.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # project root
import pandas as pd
import fuse_tools as ft

d = sys.argv[1] if len(sys.argv) > 1 else "data"
audit = pd.read_csv(f"{d}/mcu_fuse_map_audit.csv")
mcu = pd.read_csv(f"{d}/mcu_trim_fuse.csv")
m = audit.merge(mcu[["die_id", "bg_trim_code_opt", "hsi_trim_code_opt", "ldo_trim_code_opt", "ts_cal1_30C_counts"]], on="die_id")

# 1) Can we rebuild the error-free fuse words exactly from the measured/best values?
good = m[~m.has_error]
rebuilt = good.apply(lambda r: ft.build_fuse_word(r.derivative, r.bg_trim_code_opt, r.hsi_trim_code_opt,
                                                  r.ldo_trim_code_opt, r.ts_cal1_30C_counts), axis=1)
print(f"error-free words rebuilt exactly: {(rebuilt == good.fuse_map_hex.apply(lambda s: int(s, 16))).mean():.1%}")

# 2) Validate every word against the best codes, compare with the dataset's labels
def issues(r):
    return [i["type"] for i in ft.validate(r.fuse_map_hex, r.derivative,
            {"bg": r.bg_trim_code_opt, "hsi": r.hsi_trim_code_opt, "ldo": r.ldo_trim_code_opt})]
m["found"] = m.apply(issues, axis=1)
flag = m.found.apply(len) > 0
bad = m[m.has_error]
print(f"errors caught: {(flag & m.has_error).sum()}/{len(bad)} | false alarms: {(flag & ~m.has_error).sum()}/{len(good)}")
print("right error type named:", f"{bad.apply(lambda r: r.error_type in r.found, axis=1).mean():.1%}")
for t in sorted(bad.error_type.unique()):
    g = bad[bad.error_type == t]
    print(f"  {t:26s} {int(g.apply(lambda r: r.error_type in r.found, axis=1).sum())}/{len(g)}")
