from pydantic import BaseModel


class QuestionRequest(BaseModel):
    question: str


class AnswerRequest(BaseModel):
    answer: str


class SourceSelection(BaseModel):
    source_name: str


class QuestionResponse(BaseModel):
    question: str
    success: bool
    message: str


class AnswerResponse(BaseModel):
    score: str
    feedback: str
    reference_answer: str
    success: bool
    message: str


class SourceResponse(BaseModel):
    success: bool
    message: str
