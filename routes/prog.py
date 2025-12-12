from routes.schemas import base
from helpers import llm_utils

from fastapi import APIRouter, HTTPException
import os
import glob
import shutil

from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser


app = APIRouter(
    prefix="/api",
    tags=["interview"]
)

# Import global variables from rag_pipeline
from services import rag_pipeline


# API endpoint to generate an interview question
@app.get("/generate_question")
async def generate_question():
    """Generate a new interview question from the selected knowledge source"""
    if not rag_pipeline.interview_chain:
        raise HTTPException(
            status_code=400,
            detail="No knowledge source selected. Please select a source first.",
        )

    try:
        # Create a prompt for generating a single concise interview question
        interview_question_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    f"You are an expert interviewer in {rag_pipeline.current_source}. Generate exactly ONE concise interview question that would be asked in a real job interview for a {rag_pipeline.current_source} position. Make it brief (under 20 words), practical, and focused on core knowledge. Just provide the question without any additional text or explanation."
                ),
                HumanMessagePromptTemplate.from_template(
                    "Generate a brief interview question."
                ),
            ]
        )

        # Create a temporary chain for generating the interview question
        interview_question_chain = (
            interview_question_prompt | llm_utils.llm | StrOutputParser()
        )

        # Generate the interview question
        interview_question = interview_question_chain.invoke({})

        # Store the question for evaluation
        rag_pipeline.current_question = interview_question.strip()
        rag_pipeline.reference_answer = (
            "Reference answer will be generated after you submit your response."
        )

        return base.QuestionResponse(
            question=rag_pipeline.current_question,
            success=True,
            message="Question generated successfully",
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error generating question: {str(e)}"
        )