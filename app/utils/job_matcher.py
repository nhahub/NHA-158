# app/utils/job_matcher.py

from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict
import gzip
import json
import os

import requests
import faiss
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ---- Paths ----
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
COMPRESSED_JOB_INDEX_PATH = DATA_DIR / "job_embeddings.index.gz"
JOB_INDEX_PATH = DATA_DIR / "job_embeddings.index"
JOB_META_PATH = DATA_DIR / "job_chunks_metadata.json"

# Where to download the compressed index from IF it's not found locally.
# You can override this via env var JOB_INDEX_URL if needed.
DEFAULT_INDEX_URL = os.getenv("JOB_INDEX_URL") or (
    "https://raw.githubusercontent.com/nhahub/NHA-158/"
    "job_recommendation/app/data/job_embeddings.index.gz"
)


def download_compressed_index(
    url: str = DEFAULT_INDEX_URL, dest: Path = COMPRESSED_JOB_INDEX_PATH
) -> None:
    """
    Download the compressed FAISS index (.gz) from a remote URL
    into app/data/.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    print(f"[job_matcher] Downloading FAISS index from {url} ...")
    resp = requests.get(url, stream=True, timeout=600)
    resp.raise_for_status()

    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print(f"[job_matcher] Download complete: {dest}")


def ensure_index_uncompressed() -> None:
    """
    Make sure JOB_INDEX_PATH exists.

    1. If uncompressed index exists -> do nothing.
    2. Else if compressed .gz exists -> decompress it.
    3. Else -> download .gz from GitHub (or JOB_INDEX_URL) then decompress.
    """
    # 1) Already have uncompressed index
    if JOB_INDEX_PATH.exists():
        return

    # 2) If compressed file is missing, download it
    if not COMPRESSED_JOB_INDEX_PATH.exists():
        download_compressed_index()

    # 3) Decompress .gz -> .index
    print(f"[job_matcher] Decompressing {COMPRESSED_JOB_INDEX_PATH.name} ...")
    with gzip.open(COMPRESSED_JOB_INDEX_PATH, "rb") as f_in, open(
        JOB_INDEX_PATH, "wb"
    ) as f_out:
        f_out.write(f_in.read())
    print(f"[job_matcher] Decompressed to {JOB_INDEX_PATH.name}")


def load_job_index(
    index_path: Path = JOB_INDEX_PATH, metadata_path: Path = JOB_META_PATH
) -> Tuple[faiss.Index, List[Dict]]:
    """
    Load FAISS index and job chunks metadata.
    """
    # Ensure the index exists (download + decompress if needed)
    ensure_index_uncompressed()

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
