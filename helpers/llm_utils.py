from langchain_groq import ChatGroq
from . import config
import os

# Set up environment variables for API access (matching rag_api.py)
os.environ["GROQ_API_KEY"] = config.get_settings().GROQ_API_KEY
os.environ["OPENAI_API_KEY"] = config.get_settings().OPENAI_API_KEY
os.environ["LANGCHAIN_TRACING_V2"] = config.get_settings().LANGCHAIN_TRACING_V2
os.environ["LANGCHAIN_API_KEY"] = config.get_settings().LANGCHAIN_API_KEY

# Initialize the LLM
llm = ChatGroq(model="openai/gpt-oss-120b")
