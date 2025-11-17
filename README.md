---
title: Resume ATS API
emoji: 📄
colorFrom: indigo
colorTo: teal
sdk: docker
pinned: false
---

# Resume ATS API

This Space runs a FastAPI backend (Docker) that exposes:

- `GET /health` – health check  
- `POST /ats-score` – upload a PDF resume and get ATS-style scoring.

The app is started from `main.py` using `uvicorn` and a `Dockerfile`.
