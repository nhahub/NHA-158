# app/utils/job_matcher.py

from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict

import faiss
import json
import numpy as np
from dotenv import load_dotenv
load_dotenv()

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
JOB_INDEX_PATH = DATA_DIR / "job_embeddings.index"
JOB_META_PATH = DATA_DIR / "job_chunks_metadata.json"


def load_job_index(
    index_path: Path = JOB_INDEX_PATH, metadata_path: Path = JOB_META_PATH
) -> Tuple[faiss.Index, List[Dict]]:
    """
    Load FAISS index and job chunks metadata.
    """
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {index_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Job metadata JSON not found: {metadata_path}")

    index = faiss.read_index(str(index_path))
    with open(metadata_path, "r", encoding="utf-8") as f:
        meta_raw = json.load(f)

    # Some builds store {"text_to_embed":..., "metadata":{...}},
    # others store list of metadata directly. Handle both.
    job_chunks = []
    for item in meta_raw:
        if "metadata" in item:
            job_chunks.append(item)
        else:
            job_chunks.append({"text_to_embed": "", "metadata": item})

    return index, job_chunks


def match_resume_to_jobs(
    resume_embeddings: List[Dict],
    job_index: faiss.Index,
    job_chunks: List[Dict],
    resume_years: int,
    top_k: int = 5,
) -> List[Dict]:
    """
    Given resume chunk embeddings and job FAISS index, return top_k jobs.
    We:
      - search top N for each resume chunk
      - convert distance → similarity
      - aggregate best similarity per job-chunk index
      - apply experience filter: job_exp <= resume_years + 2
    """
    if not resume_embeddings:
        return []

    # convert to numpy
    resume_vecs = np.array([r["embedding"] for r in resume_embeddings], dtype="float32")

    job_scores = defaultdict(lambda: {"sim": 0.0, "max_years": 0})

    for vec in resume_vecs:
        v = np.expand_dims(vec, axis=0)
        distances, indices = job_index.search(v, k=5)

        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            # crude similarity from L2 distance
            sim = 1.0 / (1.0 + dist)

            meta = job_chunks[idx]["metadata"]
            job_exp = meta.get("max_years_exp", 0)

            # filter by experience (allow up to +2 years)
            if job_exp <= resume_years + 2:
                if sim > job_scores[idx]["sim"]:
                    job_scores[idx] = {"sim": sim, "max_years": job_exp}

    ranked = sorted(job_scores.items(), key=lambda x: x[1]["sim"], reverse=True)[:top_k]

    results = []
    for idx, info in ranked:
        meta = job_chunks[idx]["metadata"]
        results.append(
            {
                "title": meta.get("title", ""),
                "company": meta.get("company", ""),
                "location": meta.get("location", ""),
                "apply_url": meta.get("apply_url", ""),
                "required_years_experience": info["max_years"],
                "similarity": round(info["sim"], 4),
            }
        )

    return results
