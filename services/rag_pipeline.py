from helpers import llm_utils
import os
from helpers import config


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


# Global variables to store the application state
vectorstore = None
retriever = None
rag_chain = None
interview_chain = None
evaluation_chain = None
current_source = None
current_question = None
reference_answer = None


# Function to load and process data from a specific source
def load_knowledge_source(source_name):
    """
    Loads knowledge from a text file and creates embeddings for retrieval
    """
    global vectorstore, retriever, rag_chain, interview_chain, evaluation_chain, current_source

    file_path = f"data/{source_name}.pdf"

    # Check if the data file exists
    if not os.path.exists(file_path):
        return False, f"Data file {file_path} not found."

    # Document Loading
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # Text Splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    splits = text_splitter.split_documents(docs)

    # Embeddings and Vector Store
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    vectorstore = Chroma.from_documents(
        documents=splits, embedding=HuggingFaceEmbeddings(model_name=model_name)
    )

    # Create a retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    # Create RAG chain for answering questions
    system_prompt = f"You are an expert in {source_name}. Answer questions based strictly on the provided context. Include the source metadata in your response."

    prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(system_prompt),
            HumanMessagePromptTemplate.from_template("""
            Context:
            {context}

            Question:
            {question}
            """),
        ]
    )

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_template
        | llm_utils.llm
        | StrOutputParser()
    )

    # Create Interview Question Generation Chain
    interview_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                f"You are an expert in {source_name}. Based on the provided context, generate a challenging but fair 1 question that tests understanding of the material. Also provide a comprehensive reference answer. Format your response as:\n\nQUESTION: [Your question here]\nANSWER: [Your reference answer here]"
            ),
            HumanMessagePromptTemplate.from_template("""
            Context:
            {context}

            Generate a question and answer based on this context.
            """),
        ]
    )

    interview_chain = (
        {"context": retriever} | interview_prompt | llm_utils.llm | StrOutputParser()
    )

    # Create Answer Evaluation Chain
    evaluation_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template(
                f"You are an expert in {source_name}. Evaluate the user's answer compared to the reference answer. Provide a score from 0-10 and constructive feedback. Format your response as:\n\nSCORE: [0-10]\nFEEDBACK: [Your detailed feedback here]"
            ),
            HumanMessagePromptTemplate.from_template("""
            Reference Answer:
            {reference_answer}

            User Answer:
            {user_answer}

            Evaluate the user's answer.
            """),
        ]
    )

    evaluation_chain = (
        {
            "reference_answer": RunnablePassthrough(),
            "user_answer": RunnablePassthrough(),
        }
        | evaluation_prompt
        | llm_utils.llm
        | StrOutputParser()
    )

    current_source = source_name
    return True, f"{source_name} knowledge base initialized successfully!"
