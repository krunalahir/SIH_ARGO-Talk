# rag_pipeline.py
from pathlib import Path
import os
import re
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from mistralai import Mistral   # new client
from Tool import visualize_profiles, compare_profiles  # your existing functions

# -----------------------------
# Config / Data loading
# -----------------------------
BASE = Path(__file__).resolve().parent / "data"
DATA_PATH = BASE / "argo_real_sample.csv"
if not DATA_PATH.exists():
    raise FileNotFoundError(f"CSV not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
# ensure numeric types
for col in ["lat", "lon", "pressure_mean", "temperature_mean", "salinity_mean"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# -----------------------------
# Embeddings + FAISS index
# -----------------------------
embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(embed_model_name)
summaries = df["summary"].fillna("").tolist()
embeddings = embedder.encode(summaries, convert_to_numpy=True)

dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings).astype("float32"))

# Helper retriever (returns DataFrame of top_k semantic matches)
def semantic_retrieve(query: str, top_k: int = 5):
    q_emb = embedder.encode([query], convert_to_numpy=True).astype("float32")
    D, I = index.search(q_emb, top_k)
    idxs = I[0].tolist()
    matches = df.iloc[idxs].copy()
    return matches

# -----------------------------
# Mistral client
# -----------------------------
MISTRAL_KEY = os.getenv("MISTRAL_API_KEY")
if not MISTRAL_KEY:
    # Allow pipeline to run with non-LLM fallback but warn
    mistral_client = None
else:
    mistral_client = Mistral(api_key=MISTRAL_KEY)

def call_mistral(prompt: str, model: str = "mistral-small-latest"):
    """Call Mistral chat; fallback to simple echo if client missing or error."""
    if mistral_client is None:
        return "LLM not configured. Install and set MISTRAL_API_KEY."
    try:
        resp = mistral_client.chat.complete(model=model, messages=[{"role": "user", "content": prompt}])
        # defensive access
        return resp.choices[0].message.content if resp and resp.choices else ""
    except Exception as e:
        return f"LLM error: {e}"

# -----------------------------
# Intent detection & geospatial parsing
# -----------------------------
def extract_lat_lon_ranges(query: str):
    """
    Try to extract lat/lon ranges from query.
    Accepts patterns like:
      - 'between latitudes -20 to 20'
      - 'latitudes -20 to 20 and longitudes 40 to 100'
      - any sequence of 4 floats: lat_min lat_max lon_min lon_max (best-effort)
    Returns: (lat_min, lat_max), (lon_min, lon_max)
    """
    q = query.lower()
    # find all numbers (int or float)
    nums = re.findall(r"-?\d+\.\d+|-?\d+", q)
    nums = [float(n) for n in nums]
    # Attempt to locate phrases containing 'latitude' / 'longitude'
    lat_min, lat_max = -90.0, 90.0
    lon_min, lon_max = -180.0, 180.0

    # Pattern: "latitudes X to Y" or "between X and Y" near 'latitude'
    lat_match = re.search(r"lat(?:itude)?s?[^\d\-]*(-?\d+\.?\d*)\s*(?:to|and|-)\s*(-?\d+\.?\d*)", q)
    lon_match = re.search(r"lon(?:gitude)?s?[^\d\-]*(-?\d+\.?\d*)\s*(?:to|and|-)\s*(-?\d+\.?\d*)", q)
    if lat_match:
        lat_min, lat_max = sorted([float(lat_match.group(1)), float(lat_match.group(2))])
    if lon_match:
        lon_min, lon_max = sorted([float(lon_match.group(1)), float(lon_match.group(2))])

    # If explicit lat/lon not found but there are >=4 numbers, use first 4 as lat_min lat_max lon_min lon_max
    if not lat_match and not lon_match and len(nums) >= 4:
        lat_min, lat_max, lon_min, lon_max = nums[0], nums[1], nums[2], nums[3]
        # sort lat/lon ranges
        lat_min, lat_max = sorted([lat_min, lat_max])
        lon_min, lon_max = sorted([lon_min, lon_max])

    return (lat_min, lat_max), (lon_min, lon_max)

def detect_intent(query: str):
    q = query.lower()
    # priority: map -> comparison -> parameter -> text
    if any(k in q for k in ["map", "visualize", "visualise", "geospatial", "show on map", "locations", "plot"]):
        return "map"
    if any(k in q for k in ["compare", "comparison", "difference", "vs ", "versus"]):
        return "comparison"
    if any(k in q for k in ["temperature", "salinity", "oxygen", "chlorophyll", "nitrate", "parameter"]):
        return "parameter"
    return "text"

# -----------------------------
# Main API: answer_query
# -----------------------------
def answer_query(query: str, top_k: int = 5):
    """
    Returns a consistent dict:
    {
      "type": "map" | "comparison" | "parameter" | "text",
      "figure": plotly figure or None,
      "matches": pd.DataFrame (possibly empty),
      "answer": string (textual explanation)
    }
    """
    intent = detect_intent(query)
    # default result structure
    result = {"type": intent, "figure": None, "matches": pd.DataFrame(), "answer": ""}

    # ---------------- Map intent -> use geospatial filter (not semantic top-k)
    if intent == "map":
        (lat_min, lat_max), (lon_min, lon_max) = extract_lat_lon_ranges(query)
        subset = df[
            (df["lat"].between(lat_min, lat_max)) &
            (df["lon"].between(lon_min, lon_max))
        ].copy()
        result["matches"] = subset
        if subset.empty:
            result["answer"] = f"No floats found between latitudes {lat_min} to {lat_max} and longitudes {lon_min} to {lon_max}."
            result["figure"] = None
        else:
            result["answer"] = f"Showing {len(subset)} profile rows in the requested region ({lat_min},{lat_max}) x ({lon_min},{lon_max})."
            # call your tool
            try:
                result["figure"] = visualize_profiles(subset)
            except Exception as e:
                result["figure"] = None
                result["answer"] += f" (visualization error: {e})"
        return result

    # ---------------- Comparison intent -> use semantic retrieval top_k (visualize multiple profiles)
    if intent == "comparison":
        matches = semantic_retrieve(query, top_k=top_k)
        result["matches"] = matches
        if matches.empty:
            result["answer"] = "No relevant profiles found for comparison."
            return result
        # If too few rows, still attempt comparison; tool should handle it or notify
        try:
            result["figure"] = compare_profiles(matches)
            result["answer"] = f"Comparing {matches['float_id'].nunique()} floats ({len(matches)} rows) returned by semantic search."
        except Exception as e:
            result["figure"] = None
            result["answer"] = f"Could not create comparison plot: {e}"
        return result

    # ---------------- Parameter visualization -> semantic retrieve then parameter-specific plot
    if intent == "parameter":
        # choose parameter mentioned
        q = query.lower()
        param = None
        for p in ["temperature", "temp", "salinity", "oxygen", "chlorophyll", "nitrate"]:
            if p in q:
                # map common names to dataframe columns
                if p in ["temperature", "temp"]:
                    param = "temperature_mean"
                elif p == "salinity":
                    param = "salinity_mean"
                elif p == "oxygen":
                    param = "oxygen_mean" if "oxygen_mean" in df.columns else None
                elif p == "chlorophyll":
                    param = "chlorophyll_mean" if "chlorophyll_mean" in df.columns else None
                elif p == "nitrate":
                    param = "nitrate" if "nitrate" in df.columns else None
                break
        # fallback param
        if param is None:
            param = "temperature_mean" if "temperature_mean" in df.columns else df.columns[0]

        matches = semantic_retrieve(query, top_k=top_k)
        result["matches"] = matches
        if matches.empty:
            result["answer"] = "No relevant profiles found for parameter visualization."
            return result
        # create scatter param vs depth using your visualize_profiles helper (it accepts param arg)
        try:
            fig = visualize_profiles(matches, param=param)
            result["figure"] = fig
            result["answer"] = f"Parameter visualization for '{param}'."
        except Exception as e:
            result["figure"] = None
            result["answer"] = f"Could not create parameter plot: {e}"
        return result

    # ---------------- Fallback text (RAG)
    # semantic top_k retrieval used as context
    matches = semantic_retrieve(query, top_k=top_k)
    result["matches"] = matches
    context = "\n".join(matches["summary"].tolist()[:5])  # cap context
    prompt = f"You are OceanAI. Answer the query using the context below when helpful.\n\nContext:\n{context}\n\nQuery:\n{query}\n\nAnswer succinctly:"
    llm_answer = call_mistral(prompt) if mistral_client is not None else "LLM not configured."
    result["answer"] = llm_answer
    return result