# app/utils/skill_cleaner.py

import os
import re
import ast
from typing import List

import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()



def _get_gemini_model(model_name: str = "gemini-2.5-flash"):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY env var is not set.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


def clean_skills_with_gemini(raw_skills: List[str]) -> List[str]:
    """
    Use Gemini to:
      - remove noise
      - merge variants
      - normalize casing
    Returns a cleaned list of skill names.
    """
    if not raw_skills:
        return []

    model = _get_gemini_model("gemini-2.5-flash-lite")

    prompt = f"""
You are a professional resume parser.
You will receive a noisy list of extracted words or phrases from a resume.

TASK:
- Keep only real professional, technical, or soft skills.
- Remove irrelevant words (verbs, adjectives, filler terms, contact info, etc.).
- Merge duplicates and variants (e.g. "ai / ml" and "machine learning" → "Machine Learning").
- Capitalize each skill correctly.
- Return ONLY a valid Python list of the cleaned skill names.
- Do not include any explanations, code snippets, markdown, or text outside the list.
- Example of correct output:
  ["Python", "Machine Learning", "SQL", "TensorFlow", "Communication"]

Input skills:
{raw_skills}
"""

    resp = model.generate_content(prompt)
    text = (resp.text or "").strip()

    # try to keep only the list part
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        text = match.group(0)

    try:
        cleaned = ast.literal_eval(text)
    except Exception:
        cleaned = [s.strip(" -•\"'") for s in text.split(",") if s.strip()]

    # final cleanup
    cleaned = [str(s).strip() for s in cleaned if str(s).strip()]
    # dedupe preserving order
    seen = set()
    out = []
    for s in cleaned:
        key = s.lower()
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out
