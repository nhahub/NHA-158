# app/main.py

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import traceback
from traceback import print_exc

from .utils import (
    extract_text_from_pdf,
    extract_skills_from_resume,
    clean_skills_with_gemini,
    ensure_and_map,
    build_resume_embeddings,
    load_job_index,
    match_resume_to_jobs,
)

# -----------------------------------------------------------------------------
# FastAPI app
# -----------------------------------------------------------------------------
app = FastAPI(title="Resume Intelligence & Job Matching API")

# Allow frontend origins (adjust for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Change to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
class JobMatchResponse(BaseModel):
    # summary: str
    # experience_years: int
    # cleaned_skills: List[str]
    top_jobs: List[dict]


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

# Root endpoint (fixes 404 on home page)
@app.get("/")
def home():
    return {"message": "Resume Intelligence & Job Matching API is running"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/match-jobs", response_model=JobMatchResponse)
async def match_jobs_endpoint(file: UploadFile = File(...), top_k: int = 5):
    """
    Upload a resume PDF → return:
      - summary
      - years of experience
      - cleaned skills
      - top_k matching jobs
    """
    try:
    # Validate file type
        if file.content_type not in ("application/pdf", "application/octet-stream"):
            raise HTTPException(status_code=400, detail="Only PDF files are supported.")

        # Read PDF
        pdf_bytes = await file.read()

        # -------------------------------------------------------------------------
        # 1) Extract resume text
        # -------------------------------------------------------------------------
        resume_text = extract_text_from_pdf(pdf_bytes)
        if not resume_text.strip():
            raise HTTPException(status_code=400, detail="Could not extract text from PDF.")

        # -------------------------------------------------------------------------
        # 2) Extract skills with NER model
        # -------------------------------------------------------------------------
        raw_skills = extract_skills_from_resume(resume_text)

        # -------------------------------------------------------------------------
        # 3) Clean + normalize skills with Gemini
        # -------------------------------------------------------------------------
        cleaned_skills = clean_skills_with_gemini(raw_skills)

        # -------------------------------------------------------------------------
        # 4) Map to skill ontology (updates CSV if needed)
        # -------------------------------------------------------------------------
        mapped_skills = ensure_and_map(cleaned_skills)

        # -------------------------------------------------------------------------
        # 5) Build resume embeddings (summary + skills + years)
        # -------------------------------------------------------------------------
        combined_resume, resume_embs = build_resume_embeddings(
            resume_text=resume_text,
            mapped_skills=mapped_skills
        )

        # -------------------------------------------------------------------------
        # 6) Load FAISS job index + metadata & find top matches
        # -------------------------------------------------------------------------
        job_index, job_chunks = load_job_index()

        top_jobs = match_resume_to_jobs(
            resume_embeddings=resume_embs,
            job_index=job_index,
            job_chunks=job_chunks,
            resume_years=combined_resume["experience_years"],
            top_k=top_k,
        )

    # -------------------------------------------------------------------------
    # Return final response
    # -------------------------------------------------------------------------
        return JobMatchResponse(
            # summary=combined_resume["summary"],
            # experience_years=combined_resume["experience_years"],
            # cleaned_skills=[s["preferred_label"] for s in mapped_skills],
            top_jobs=top_jobs,
        )
    
    except Exception as e:
        print("\n🔥 SERVER ERROR 🔥")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
