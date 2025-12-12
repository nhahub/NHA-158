from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from routes import prog, user
from helpers import config
import os

# Initialize FastAPI app
app = FastAPI(
    title="Interview Q&A API", description="API for interview preparation and Q&A"
)

# Add CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the routers
app.include_router(prog.app)


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Interview Q&A API",
        "version": "1.0.0",
        "endpoints": {
            "sources": "/api/sources - Get list of available knowledge sources",
            "select_source": "/api/select_source - Select a knowledge source",
            "generate_question": "/api/generate_question - Generate an interview question",
            "submit_answer": "/api/submit_answer - Submit an answer for evaluation",
        },
    }


# Run the app
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
