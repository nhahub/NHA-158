# app/utils/resume_parser.py

from typing import List, Union
from PyPDF2 import PdfReader
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from dotenv import load_dotenv
load_dotenv()

# Hugging Face model repo
NER_MODEL_NAME = "hagarrrr/bert_skill_ner_best"  # Replace with your Hugging Face repo

_ner_pipeline = None  # cached pipeline

def get_ner_pipeline():
    """
    Lazy-load and cache the NER pipeline so we don't reload it every request.
    Downloads the model from Hugging Face Hub if not already cached.
    """
    global _ner_pipeline
    if _ner_pipeline is None:
        tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_NAME)
        model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_NAME)
        _ner_pipeline = pipeline(
            "ner",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy="simple",
        )
    return _ner_pipeline


def extract_text_from_pdf(pdf_source: Union[str, bytes]) -> str:
    """
    Extract plain text from a PDF file.
    - pdf_source can be a file path or raw bytes.
    """
    if isinstance(pdf_source, str):
        reader = PdfReader(pdf_source)
    else:
        from io import BytesIO
        reader = PdfReader(BytesIO(pdf_source))

    pages_text = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages_text).strip()


def extract_skills_from_resume(text: str) -> List[str]:
    """
    Run the fine-tuned BERT NER model over resume text and return unique SKILL entities.
    """
    nlp = get_ner_pipeline()
    ner_results = nlp(text)

    skills = sorted(
        {r["word"].strip() for r in ner_results if r.get("entity_group") == "SKILL"}
    )
    return skills
