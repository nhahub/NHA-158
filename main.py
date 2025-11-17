import os
import re
import json
import io
from pathlib import Path
from typing import List, Dict, Any, Tuple

import pdfplumber
import google.generativeai as genai
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()


# ========= Config: Gemini API =========
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY environment variable is not set. "
        "Set it before running: e.g. $env:GEMINI_API_KEY='YOUR_KEY'"
    )

genai.configure(api_key=GEMINI_API_KEY)
MODEL_NAME = "gemini-2.5-flash-lite"

# ========= FastAPI app + CORS =========
app = FastAPI(
    title="Resume ATS Scoring API",
    description="Upload a PDF resume and get ATS-style scoring using Gemini 2.5.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later restrict to your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========= Prompt & helpers =========
VISUAL_AWARE_PROMPT = """
You are an ATS auditor. You receive:
1️⃣ RESUME_TEXT: raw plaintext extracted from a PDF résumé (may have broken spacing/wrapping).
2️⃣ VISUAL_SUMMARY: a machine-generated summary of layout features from the PDF:
   - page_count, urls, emails, phones
   - bullets_total and per-page breakdown
   - headings (text, size, boldness)
   - section_candidates (likely section titles)
   - columns_guess (single/dual)

Instructions:
- Use BOTH text and visuals.
- If either text or visuals show LinkedIn, email, phone, or sections, don’t mark them as missing.
- Normalize common PDF artifacts (spaced letters, wrapped lines) when judging structure.
- Mark something missing ONLY if both views suggest absence.
- Prefer objective signals (contact info, bullets, section headers, quantified results).
- Consider presence of action verbs and metrics for impact.

Scoring (0–100):
• format: layout quality (sections, bullets, alignment)
• parseability: ATS-friendliness (plain text, delimiters)
• structure: presence/order of main sections
• content: action verbs, quantifiable results, skills breadth
• readability: concise, consistent, well-structured text
• hygiene: typos, broken links/emails, punctuation

Overall = round(0.25*format + 0.20*parseability + 0.20*structure + 0.20*content + 0.10*readability + 0.05*hygiene).

Output STRICT JSON ONLY:
{
  "overall": <int>,
  "subscores": {
    "format": <int>,
    "parseability": <int>,
    "structure": <int>,
    "content": <int>,
    "readability": <int>,
    "hygiene": <int>
  },
  "warnings": ["string", ...],
  "missing_or_weak_sections": ["string", ...],
  "top_recommendations": ["concise actionable item", ...]
}
""".strip()


def robust_json_load(x: str) -> Dict[str, Any]:
    x = (x or "").strip()
    try:
        return json.loads(x)
    except Exception:
        s, e = x.find("{"), x.rfind("}")
        if s != -1 and e != -1 and e > s:
            return json.loads(x[s:e+1])
        raise


def ats_score_with_llm_visual(resume_text: str, visual_summary: Dict[str, Any]) -> Dict[str, Any]:
    model = genai.GenerativeModel(MODEL_NAME)

    prompt = f"""System:
{VISUAL_AWARE_PROMPT}

User:
RESUME_TEXT:
\"\"\"{resume_text[:150000]}\"\"\"  # truncated if huge

VISUAL_SUMMARY (JSON):
{json.dumps(visual_summary, ensure_ascii=False, indent=2)}

Return only the JSON described above.
"""
    resp = model.generate_content(prompt)
    return robust_json_load(resp.text)


# ========= PDF parsing & visual summary =========

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
URL_RE = re.compile(r"(https?://[^\s]+)")
PHONE_RE = re.compile(r"(\+?\d[\d\s\-()]{7,})")


def extract_resume_text_and_visual(pdf_bytes: bytes) -> Tuple[str, Dict[str, Any]]:
    text_per_page: List[str] = []
    bullets_per_page: List[int] = []

    # pdfplumber.open expects a file-like object, so wrap bytes
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text_per_page.append(page_text)

            bullet_count = 0
            for line in page_text.splitlines():
                stripped = line.strip()
                if stripped.startswith(("•", "-", "·", "*")):
                    bullet_count += 1
            bullets_per_page.append(bullet_count)

    full_text = "\n".join(text_per_page)
    page_count = len(text_per_page)

    emails = list(set(EMAIL_RE.findall(full_text)))
    urls = list(set(URL_RE.findall(full_text)))
    phones = list(set(PHONE_RE.findall(full_text)))

    section_candidates: List[str] = []
    for line in full_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if (
            len(stripped) <= 50
            and stripped.isupper()
            and any(keyword in stripped for keyword in ["SUMMARY", "EXPERIENCE", "EDUCATION", "SKILLS", "PROJECT"])
        ):
            section_candidates.append(stripped)

    visual_summary = {
        "page_count": page_count,
        "urls": urls,
        "emails": emails,
        "phones": phones,
        "bullets_total": int(sum(bullets_per_page)),
        "bullets_per_page": bullets_per_page,
        "headings": [],
        "section_candidates": section_candidates,
        "columns_guess": "single",
    }

    return full_text, visual_summary


# ========= Pydantic models =========

class ATSSubscores(BaseModel):
    format: int
    parseability: int
    structure: int
    content: int
    readability: int
    hygiene: int


class ATSResponse(BaseModel):
    overall: int
    subscores: ATSSubscores
    warnings: list[str] = []
    missing_or_weak_sections: list[str] = []
    top_recommendations: list[str] = []
    visual_summary: Dict[str, Any]


# ========= Endpoints =========

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ats-score", response_model=ATSResponse)
async def ats_score_endpoint(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Uploaded file must be a PDF.")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty file uploaded.")

    try:
        resume_text, visual_summary = extract_resume_text_and_visual(pdf_bytes)

        if not resume_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from the PDF.")

        ats_result = ats_score_with_llm_visual(resume_text, visual_summary)

        return ATSResponse(
            overall=int(ats_result.get("overall", 0)),
            subscores=ATSSubscores(
                format=int(ats_result.get("subscores", {}).get("format", 0)),
                parseability=int(ats_result.get("subscores", {}).get("parseability", 0)),
                structure=int(ats_result.get("subscores", {}).get("structure", 0)),
                content=int(ats_result.get("subscores", {}).get("content", 0)),
                readability=int(ats_result.get("subscores", {}).get("readability", 0)),
                hygiene=int(ats_result.get("subscores", {}).get("hygiene", 0)),
            ),
            warnings=ats_result.get("warnings", []),
            missing_or_weak_sections=ats_result.get("missing_or_weak_sections", []),
            top_recommendations=ats_result.get("top_recommendations", []),
            visual_summary=visual_summary,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=7860, reload=True)
