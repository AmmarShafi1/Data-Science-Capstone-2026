from pathlib import Path
import re
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

ONET_DIR = Path("onet_db")
OUT_CSV = "soc_ai_similarity.csv"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOP_K = 5  # average of top-K row similarities per SOC

AI_ANCHOR_TEXT = """
artificial intelligence systems
machine learning models
predictive analytics software
decision support systems
recommendation systems
automation of cognitive tasks
intelligent software tools
AI-powered applications
model outputs interpretation
human-in-the-loop AI
algorithmic decision making
data-driven automation
AI workflow integration
AI system configuration
AI-assisted analysis
AI tooling
"""

AUTOMATION_ANCHOR_TEXT = """
industrial automation manufacturing assembly line production process machine operation
equipment monitoring numerical control cnc plc scada control systems plant operations
quality inspection conveyor packaging machining welding forklift warehouse
"""

SOC_PATTERN = re.compile(r"^\d{2}-\d{4}\.\d{2}$|^\d{2}-\d{4}$")

def read_onet_table(path: Path) -> pd.DataFrame:
    for sep in ["\t", ","]:
        try:
            df = pd.read_csv(path, sep=sep, dtype=str, engine="python")
            if df.shape[1] >= 2:
                df.columns = [c.strip() for c in df.columns]
                return df
        except Exception:
            pass
    raise RuntimeError(f"Could not parse {path.name}")

def find_soc_col(df: pd.DataFrame) -> str:
    for c in ["O*NET-SOC Code","ONET-SOC Code","O_NET-SOC Code"]:
        if c in df.columns:
            return c
    for c in df.columns:
        if "SOC" in c.upper() and "CODE" in c.upper():
            return c
    raise RuntimeError("SOC code column not found")

def normalize_soc_to_2018(onet_soc: str) -> str:
    return str(onet_soc).split(".")[0].strip()

def clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()

def load_titles() -> pd.DataFrame:
    f = ONET_DIR / "Occupation Data.txt"
    if not f.exists():
        return pd.DataFrame(columns=["onet_soc_code","title"])
    df = read_onet_table(f)
    soc_col = find_soc_col(df)
    title_col = "Title" if "Title" in df.columns else None
    if not title_col:
        return pd.DataFrame(columns=["onet_soc_code","title"])
    out = df[[soc_col, title_col]].copy()
    out.columns = ["onet_soc_code","title"]
    out["onet_soc_code"] = out["onet_soc_code"].astype(str).str.strip()
    out["title"] = out["title"].astype(str).str.strip()
    return out.drop_duplicates("onet_soc_code")

# ---- Load Technology Skills table (row-level) ----
tech_file = ONET_DIR / "Technology Skills.txt"
if not tech_file.exists():
    raise RuntimeError("Missing Technology Skills.txt in onet_db/")

tech = read_onet_table(tech_file)
soc_col = find_soc_col(tech)

# pick best available text columns for tech skills
text_cols = [c for c in ["Technology Skill","Title","Commodity Title","Description","Example"] if c in tech.columns]
if not text_cols:
    # fallback to non-id columns
    text_cols = [c for c in tech.columns if "id" not in c.lower() and "code" not in c.lower()][:4]

tech = tech[[soc_col] + text_cols].copy()
tech = tech.rename(columns={soc_col: "onet_soc_code"})
tech["onet_soc_code"] = tech["onet_soc_code"].astype(str).str.strip()
tech = tech[tech["onet_soc_code"].apply(lambda x: bool(SOC_PATTERN.match(x)))]

tech["row_text"] = tech[text_cols].astype(str).agg(" | ".join, axis=1).map(clean_text)
tech = tech[tech["row_text"].str.len() > 0].reset_index(drop=True)

# ---- Embed anchors + rows ----
model = SentenceTransformer(MODEL_NAME)
ai_emb = model.encode([AI_ANCHOR_TEXT], normalize_embeddings=True)
auto_emb = model.encode([AUTOMATION_ANCHOR_TEXT], normalize_embeddings=True)

row_emb = model.encode(tech["row_text"].tolist(), normalize_embeddings=True, show_progress_bar=True)

ai_sim = cosine_similarity(row_emb, ai_emb).ravel()
auto_sim = cosine_similarity(row_emb, auto_emb).ravel()

tech["ai_sim_raw"] = ai_sim
tech["automation_sim_raw"] = auto_sim
tech["ai_exposure_row"] = tech["ai_sim_raw"] - np.maximum(tech["automation_sim_raw"], 0.0)

# ---- Aggregate per SOC using top-K mean ----
def topk_mean(x: pd.Series, k: int) -> float:
    if len(x) == 0:
        return np.nan
    return float(np.mean(np.sort(x.values)[-min(k, len(x)):]))

soc = tech.groupby("onet_soc_code").agg(
    ai_exposure=("ai_exposure_row", lambda s: topk_mean(s, TOP_K)),
    ai_sim_raw=("ai_sim_raw", lambda s: topk_mean(s, TOP_K)),
    automation_sim_raw=("automation_sim_raw", lambda s: topk_mean(s, TOP_K)),
    n_rows_used=("ai_exposure_row", "size")
).reset_index()

soc["soc_2018"] = soc["onet_soc_code"].map(normalize_soc_to_2018)

titles = load_titles()
soc = soc.merge(titles, on="onet_soc_code", how="left")
soc["title"] = soc["title"].fillna("Unknown")

# optional: text_len not meaningful now (row-level), keep as NA
soc["text_len"] = np.nan

soc = soc.sort_values("ai_exposure", ascending=False)
soc.to_csv(OUT_CSV, index=False)

print("Wrote", OUT_CSV)
print(soc[["onet_soc_code","title","ai_exposure","ai_sim_raw","automation_sim_raw"]].head(20).to_string(index=False))
