import pandas as pd
import numpy as np
import re
from pathlib import Path

# =========================
# CONFIG — SET YOUR FILES
# =========================
SOC_AI_PATH = Path("soc_ai_similarity.csv")               # your SOC AI file
CES_WAGES_PATH = Path("ces_hourly_earnings_2022_2025.csv")# your CES monthly wages file

# IMPORTANT: this should be an OEWS file with NAICS + SOC + employment
# Examples: national "industry x occupation" tables you downloaded from OEWS
OEWS_PATH = Path("all_data_M_2024.csv")

# Column names in your OEWS file (edit these to match your file)
OEWS_NAICS_COL = "NAICS"          # e.g., "naics", "naics_code", "NAICS"
OEWS_SOC_COL   = "OCC_CODE"       # e.g., "occ_code", "soc", "OCC_CODE"
OEWS_EMP_COL   = "TOT_EMP"        # e.g., "tot_emp", "employment", "TOT_EMP"

# Which SOC score column to use from soc_ai_similarity.csv
SOC_SCORE_COL = "ai_exposure"     # or whatever your final tooling-familiarity score column is

# Output
OUT_AI_BY_NAICS = Path("ai_intensity_by_naics2.csv")
OUT_AI_BY_CES   = Path("ai_intensity_by_ces_bucket.csv")
OUT_PANEL       = Path("ces_wages_with_ai_intensity.csv")

# =========================
# CES SUPERSECTOR ↔ NAICS MAPPING
# =========================
# Your CES file likely has "industry_bucket" like: "education_health", "information", etc.
# Map each CES bucket to a set of 2-digit NAICS sectors.
CES_BUCKET_TO_NAICS2 = {
    # Common CES supersectors
    "information": {"51"},
    "financial_activities": {"52", "53"},
    "professional_business_services": {"54", "55", "56"},
    "education_health": {"61", "62"},
    "leisure_hospitality": {"71", "72"},
    "trade_transportation_utilities": {"42", "44", "45", "48", "49", "22"},
    "manufacturing": {"31", "32", "33"},
    "construction": {"23"},
    "mining_logging": {"21"},
    "other_services": {"81"},

    # If you have these in CES
    "government": {"92"},  # (OEWS often uses 92 for public admin; sometimes government is treated differently)
    # Total private is "all private industries" => everything except government-ish.
    # We'll compute it as the employment-weighted mean across all NAICS2 in your OEWS.
    "total_private": None,
}

# =========================
# HELPERS
# =========================
def normalize_soc(s: str) -> str:
    """Normalize SOC strings like '15-1252.00' -> '15-1252'."""
    if pd.isna(s):
        return None
    s = str(s).strip()
    return s.split(".")[0]

def normalize_naics2(n: str) -> str:
    """Take NAICS code and return first 2 digits when possible."""
    if pd.isna(n):
        return None
    n = str(n).strip()
    # OEWS NAICS codes may be like '5415', '62', '622110', '31-33', etc.
    # Handle ranges like '31-33' by returning None (you can expand this if needed).
    if "-" in n:
        return None
    # Keep only digits
    digits = re.sub(r"\D", "", n)
    if len(digits) >= 2:
        return digits[:2]
    return None

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return np.nan

# =========================
# 1) LOAD SOC AI SCORES
# =========================
soc = pd.read_csv(SOC_AI_PATH)

# Prefer soc_2018 if present; else fall back to onet_soc_code normalized
if "soc_2018" in soc.columns:
    soc["soc_code"] = soc["soc_2018"].astype(str).map(normalize_soc)
else:
    soc["soc_code"] = soc["onet_soc_code"].astype(str).map(normalize_soc)

if SOC_SCORE_COL not in soc.columns:
    raise ValueError(f"Expected SOC score column '{SOC_SCORE_COL}' not found in {SOC_AI_PATH}.\n"
                     f"Columns: {list(soc.columns)}")

soc = soc[["soc_code", SOC_SCORE_COL]].dropna()
soc[SOC_SCORE_COL] = soc[SOC_SCORE_COL].map(safe_float)
soc = soc.dropna()

# Drop aggregate SOCs if present (e.g., 00-0000)
soc = soc[soc["soc_code"].str.match(r"^\d{2}-\d{4}$")]

# =========================
# 2) LOAD OEWS SOC×NAICS EMPLOYMENT
# =========================
oews = pd.read_csv(OEWS_PATH, dtype=str)

for col in [OEWS_NAICS_COL, OEWS_SOC_COL, OEWS_EMP_COL]:
    if col not in oews.columns:
        raise ValueError(f"OEWS file missing required column '{col}'. "
                         f"Found columns: {list(oews.columns)}")

oews["soc_code"] = oews[OEWS_SOC_COL].map(normalize_soc)
oews["naics2"] = oews[OEWS_NAICS_COL].map(normalize_naics2)
oews["emp"] = oews[OEWS_EMP_COL].map(safe_float)

# Clean
oews = oews.dropna(subset=["soc_code", "naics2", "emp"])
oews = oews[oews["emp"] > 0]
oews = oews[oews["soc_code"].str.match(r"^\d{2}-\d{4}$")]

# =========================
# 3) MERGE SOC AI INTO OEWS AND COMPUTE AI INTENSITY BY NAICS2
# =========================
m = oews.merge(soc, on="soc_code", how="inner")

# Weighted average AI score by NAICS2
ai_by_naics2 = (
    m.groupby("naics2")
     .apply(lambda g: np.average(g[SOC_SCORE_COL], weights=g["emp"]))
     .reset_index(name="ai_intensity")
)

# Add total employment by NAICS2 for transparency
emp_by_naics2 = m.groupby("naics2")["emp"].sum().reset_index(name="emp_total")
ai_by_naics2 = ai_by_naics2.merge(emp_by_naics2, on="naics2", how="left")

ai_by_naics2.to_csv(OUT_AI_BY_NAICS, index=False)
print(f"Wrote {OUT_AI_BY_NAICS} ({len(ai_by_naics2)} NAICS2 sectors)")

# =========================
# 4) AGGREGATE NAICS2 → CES BUCKET AI INTENSITY
# =========================
# Build a lookup for NAICS2 -> ai_intensity and NAICS2 -> emp_total
ai_lookup = ai_by_naics2.set_index("naics2")["ai_intensity"].to_dict()
emp_lookup = ai_by_naics2.set_index("naics2")["emp_total"].to_dict()

def bucket_ai(bucket: str):
    if bucket == "total_private":
        # employment-weighted mean across all NAICS2 we have
        naics_list = list(ai_lookup.keys())
    else:
        naics_set = CES_BUCKET_TO_NAICS2.get(bucket)
        if naics_set is None:
            return np.nan
        naics_list = [n for n in naics_set if n in ai_lookup]

    if not naics_list:
        return np.nan

    weights = np.array([emp_lookup[n] for n in naics_list], dtype=float)
    values = np.array([ai_lookup[n] for n in naics_list], dtype=float)
    if weights.sum() == 0:
        return np.nan
    return float(np.average(values, weights=weights))

ai_by_ces = pd.DataFrame({
    "industry_bucket": list(CES_BUCKET_TO_NAICS2.keys()),
})
ai_by_ces["ai_intensity"] = ai_by_ces["industry_bucket"].map(bucket_ai)

ai_by_ces.to_csv(OUT_AI_BY_CES, index=False)
print(f"Wrote {OUT_AI_BY_CES} ({len(ai_by_ces)} CES buckets)")

# =========================
# 5) MERGE INTO CES MONTHLY WAGES
# =========================
ces = pd.read_csv(CES_WAGES_PATH)

# Expect something like: industry_bucket, date, avg_hourly_earnings (your earlier format)
needed = {"industry_bucket", "avg_hourly_earnings"}
missing = needed - set(ces.columns)
if missing:
    raise ValueError(f"CES file missing expected columns: {missing}. "
                     f"Found columns: {list(ces.columns)}")

panel = ces.merge(ai_by_ces, on="industry_bucket", how="left")

panel.to_csv(OUT_PANEL, index=False)
print(f"Wrote {OUT_PANEL} ({len(panel):,} rows)")

print("\nPreview:")
print(panel.head(10).to_string(index=False))
