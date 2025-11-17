# app/utils/embeddings.py

from typing import List, Dict, Tuple

import google.generativeai as genai
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()


_emb_model = None  # cached sentence-transformers model


def get_embedding_model():
    global _emb_model
    if _emb_model is None:
        _emb_model = SentenceTransformer("all-mpnet-base-v2")
    return _emb_model


def _get_gemini_model(model_name: str = "gemini-2.5-flash"):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY env var is not set.")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_name)


def extract_resume_summary_experience(resume_text: str) -> Dict:
    """
    Use Gemini to extract:
      - short professional summary
      - total years of professional experience
    """
    SYSTEM_PROMPT = """
You are an expert resume analyzer.
Extract from the resume text:
1. A concise professional summary (1-4 sentences)
2. Total years of professional experience, counting only paid jobs or official internships.

Return STRICT JSON ONLY with keys: summary, experience_years
""".strip()

    model = _get_gemini_model("gemini-2.5-flash")
    resp = model.generate_content(f"{SYSTEM_PROMPT}\n\nResume Text:\n{resume_text}")
    text = (resp.text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        import re

        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return json.loads(m.group(0))
        raise ValueError("Gemini did not return valid JSON for summary/experience")


def build_resume_chunks(
    summary: str, experience_years: float, mapped_skills: List[Dict]
) -> List[Dict]:
    """
    Convert summary + mapped skills into chunk objects ready for embedding.
    Each chunk:
      - chunk_id
      - text_to_embed
      - metadata
    """
    chunks = []

    chunks.append(
        {
            "chunk_id": "resume_summary",
            "section": "summary",
            "text_to_embed": f"Summary: {summary}. Years of Experience: {experience_years}",
            "metadata": {
                "section": "summary",
                "experience_years": experience_years,
            },
        }
    )

    chunk_size = 8
    for i in range(0, len(mapped_skills), chunk_size):
        skills_chunk = mapped_skills[i : i + chunk_size]
        skills_text = " ".join(
            f"Skill: {s['preferred_label']}. Also called: {', '.join(s.get('alt_labels', []))}. "
            f"Broader category: {s.get('broader', '')}. Related skills: {', '.join(s.get('related', []))}."
            for s in skills_chunk
        )
        chunks.append(
            {
                "chunk_id": f"resume_chunk_{i // chunk_size + 1}",
                "section": "skills",
                "text_to_embed": skills_text.strip(),
                "metadata": {
                    "section": "skills",
                    "experience_years": experience_years,
                },
            }
        )

    return chunks


def embed_chunks(chunks: List[Dict]) -> List[Dict]:
    model = get_embedding_model()
    out = []
    for ch in chunks:
        vec = model.encode(ch["text_to_embed"]).tolist()
        out.append(
            {
                "chunk_id": ch["chunk_id"],
                "embedding": vec,
                "metadata": ch["metadata"],
            }
        )
    return out


def build_resume_embeddings(
    resume_text: str, mapped_skills: List[Dict]
) -> Tuple[Dict, List[Dict]]:
    """
    Full pipeline:
      - extract summary & years of experience
      - build chunks
      - embed chunks
    Returns:
      combined_json_resume, resume_embeddings
    """
    info = extract_resume_summary_experience(resume_text)
    summary = info.get("summary", "")
    years = info.get("experience_years", 0.0)

    chunks = build_resume_chunks(summary, years, mapped_skills)
    embeddings = embed_chunks(chunks)

    combined_resume = {
        "summary": summary,
        "experience_years": years,
        "skills": mapped_skills,
    }

    return combined_resume, embeddings
