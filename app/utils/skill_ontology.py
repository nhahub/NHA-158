# app/utils/skill_ontology.py

import os
import json
import shutil
from pathlib import Path
from typing import List, Dict

import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

# Paths
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CSV_PATH = DATA_DIR / "custom_skill_ontology (2) (4).csv"

GEMINI_MODEL = "gemini-2.5-flash-lite"

EXPECTED_COLS = ["skill_id", "preferred_label", "alt_labels", "broader", "related"]


def _ensure_csv_exists():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0:
        df = pd.DataFrame(columns=EXPECTED_COLS)
        df.to_csv(CSV_PATH, index=False, encoding="utf-8")
        return df

    df = pd.read_csv(CSV_PATH)
    if list(df.columns) != EXPECTED_COLS:
        backup = CSV_PATH.with_suffix(".bak")
        df.to_csv(backup, index=False)
        df = pd.DataFrame(columns=EXPECTED_COLS)
        df.to_csv(CSV_PATH, index=False, encoding="utf-8")
    return df


SYSTEM_PROMPT = """You are an expert skills taxonomist for AI-powered ATS systems.
You will receive raw skill mentions extracted from resumes and job descriptions.
Normalize both technical and soft skills into a unified ontology.

Output STRICT JSON only (no prose).
Each record must contain these keys:
skill_id, preferred_label, alt_labels, broader, related.

Rules:
- Deduplicate synonyms/variants into ONE canonical record (use alt_labels).
- Each skill must have AT LEAST 3 realistic alt_labels (synonyms, abbreviations, or variations).
  Examples:
    - "Python" → ["Python3", "Python programming", "Python language"]
    - "Docker" → ["Docker Engine", "Docker tool", "Containerization"]
- Assign exactly ONE broader category from the allowed list.
- Provide between 3 and 6 realistic related skills (peers/complements; NOT the parent).
- Use standard casing (e.g., "Python", "Scikit-learn", "Power BI", "Teamwork").
- Only include widely recognized skills/tools (no inventions or obscure terms).

Allowed broader categories:
["Programming Languages","ML Frameworks","ML Algorithms","ML Subfields","AI Subfields",
 "NLP Libraries","Computer Vision Libraries","Data Analysis","Data Visualization",
 "Data Engineering","Big Data","Databases","BI Tools","Web Technologies","Web Frameworks",
 "Frontend Frameworks","Backend Runtimes","APIs","Auth","DevOps","Operating Systems","Security",
 "Soft Skills","Communication","Leadership","Problem Solving","Management","Creativity",
 "Collaboration","Time Management","Adaptability","Analytical Thinking"]
""".strip()

USER_TEMPLATE = """Raw skill mentions (one per line):
{skills_block}

== Output format ==
Return a JSON ARRAY where each element follows this schema:
{{
  "skill_id": "s###",
  "preferred_label": "string",
  "alt_labels": ["string", "..."],
  "broader": "<one allowed category>",
  "related": ["string", "..."]
}}""".strip()


def build_synonym_map(df: pd.DataFrame) -> Dict[str, str]:
    syn = {}
    if df.empty:
        return syn
    for _, r in df.iterrows():
        pref = str(r.get("preferred_label", "")).strip()
        if pref:
            syn[pref.lower()] = pref
        alts = str(r.get("alt_labels", "")).split(";")
        for a in alts:
            a = a.strip()
            if a:
                syn[a.lower()] = pref
    return syn


def _robust_json_load(text: str):
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        start, end = text.find("["), text.rfind("]")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise ValueError("Model returned non-JSON:\n" + text[:600])


def _get_gemini_model():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY env var is not set.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(GEMINI_MODEL)


def call_gemini_ontology(skills: List[str]) -> List[dict]:
    if not skills:
        return []
    model = _get_gemini_model()
    block = "\n".join(sorted(set(skills)))
    prompt = f"{SYSTEM_PROMPT}\n\n{USER_TEMPLATE.format(skills_block=block)}"
    resp = model.generate_content(prompt)
    return _robust_json_load(resp.text)


def normalize_new_records(new_records: List[dict]) -> pd.DataFrame:
    rows = []
    for r in new_records:
        rows.append(
            {
                "skill_id": r.get("skill_id", ""),
                "preferred_label": r.get("preferred_label", "").strip(),
                "alt_labels": "; ".join(
                    [
                        str(x).strip()
                        for x in r.get("alt_labels", [])
                        if str(x).strip().lower() != "nan"
                    ]
                ),
                "broader": r.get("broader", "").strip(),
                "related": ", ".join(
                    [str(x).strip() for x in r.get("related", []) if str(x).strip()]
                ),
            }
        )
    return pd.DataFrame(rows, columns=EXPECTED_COLS)


def get_skill_metadata(extracted_skills: List[str], ontology_path: Path = CSV_PATH):
    df = pd.read_csv(ontology_path)

    synmap = {}
    for _, r in df.iterrows():
        pref = str(r["preferred_label"]).strip()
        if pref:
            synmap[pref.lower()] = pref
        alts = str(r.get("alt_labels", "")).split(";")
        for alt in alts:
            alt = alt.strip()
            if alt:
                synmap[alt.lower()] = pref

    mapped = []
    for s in extracted_skills:
        k = s.lower().strip()
        if k in synmap:
            mapped.append(synmap[k])

    df_subset = df[df["preferred_label"].isin(mapped)].copy()

    results = []
    for _, r in df_subset.iterrows():
        results.append(
            {
                "skill_id": r.get("skill_id", ""),
                "preferred_label": r["preferred_label"],
                "alt_labels": [
                    a.strip() for a in str(r["alt_labels"]).split(";") if a.strip()
                ],
                "broader": r["broader"],
                "related": [
                    rel.strip()
                    for rel in str(r["related"]).split(",")
                    if rel.strip()
                ],
            }
        )
    return results


def ensure_and_map(skills: List[str]) -> List[dict]:
    """
    Ensure all skills exist in the ontology (create missing ones with Gemini),
    update the CSV, and return ontology metadata rows corresponding to the
    given skills.
    """
    df = _ensure_csv_exists()

    synmap = build_synonym_map(df)
    missing = sorted({s for s in skills if s.lower() not in synmap})

    if missing:
        new_records = call_gemini_ontology(missing)
        if new_records:
            df_new = normalize_new_records(new_records)
            df = pd.concat([df, df_new], ignore_index=True)
            df = df.drop_duplicates(subset=["preferred_label"], keep="last").reset_index(
                drop=True
            )

            if CSV_PATH.exists() and CSV_PATH.stat().st_size > 0:
                shutil.copy(CSV_PATH, CSV_PATH.with_suffix(".bak"))
            df.to_csv(CSV_PATH, index=False, encoding="utf-8")

    return get_skill_metadata(skills, ontology_path=CSV_PATH)
