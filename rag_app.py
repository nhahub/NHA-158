# Import necessary libraries for the RAG application
import streamlit as st  # For creating the web interface
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
)  # For loading documents from files
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)  # For splitting large documents into chunks
from langchain_community.vectorstores import (
    Chroma,
)  # For storing and retrieving vector embeddings
from langchain_community.embeddings import (
    HuggingFaceEmbeddings,
)  # For creating text embeddings
from langchain_groq import ChatGroq  # For accessing the Groq LLM API
from langchain_core.prompts import (
    ChatPromptTemplate,  # For creating structured prompts for the LLM
    SystemMessagePromptTemplate,  # For system-level instructions
    HumanMessagePromptTemplate,  # For user-level instructions
)
from langchain_core.output_parsers import StrOutputParser  # For parsing LLM responses
from langchain_core.runnables import (
    RunnablePassthrough,
)  # For passing data through the chain
import os  # For accessing environment variables

# Set up environment variables for API access
os.environ["GROQ_API_KEY"] = ""  # Groq API key for LLM access
os.environ["OPENAI_API_KEY"] = ""  # OpenAI API key (backup)
os.environ["LANGCHAIN_TRACING_V2"] = "true"  # Enable LangChain tracing for debugging
os.environ["LANGCHAIN_API_KEY"] = ""  # LangSmith API key for tracing

# App title and description
st.title("Interview_Q&A_ChatBot")  # Main title of the application
st.write("First select a role from below options")  # Brief description of the app

# Initialize session state variables to persist data across user interactions
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = (
        None  # Stores the vector database for similarity search
    )
if "retriever" not in st.session_state:
    st.session_state.retriever = None  # Retrieves relevant documents from vector store
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = (
        None  # Chain for answering questions (not used in quiz mode)
    )
if "current_source" not in st.session_state:
    st.session_state.current_source = None  # Tracks which knowledge source is selected
if "Interview_mode" not in st.session_state:
    st.session_state.Interview_mode = False  # Flag to indicate quiz mode is active
if "current_question" not in st.session_state:
    st.session_state.current_question = None  # Stores the current quiz question
if "reference_answer" not in st.session_state:
    st.session_state.reference_answer = (
        None  # Stores the correct answer for current question
    )
if "Interview_chain" not in st.session_state:
    st.session_state.Interview_chain = None  # Chain for generating quiz questions
if "evaluation_chain" not in st.session_state:
    st.session_state.evaluation_chain = None  # Chain for evaluating user answers


# Function to load and process data from a specific source
def load_knowledge_source(source_name):
    """
    Loads knowledge from a text file and creates embeddings for retrieval

    Args:
        source_name (str): Name of the knowledge source (AI, SQL, or PySpark)

    Returns:
        bool: True if loading was successful, False otherwise
    """
    file_path = f"data/{source_name}.pdf"  # Path to the knowledge source file

    # Check if the data file exists before proceeding
    if not os.path.exists(file_path):
        st.error(f"Data file {file_path} not found.")  # Display error message to user
        return False  # Return False to indicate failure

    try:
        # Show loading spinner while processing documents
        with st.spinner(f"Loading {source_name} documents and creating embeddings..."):
            # Document Loading - Load text from the specified file
            loader = PyPDFLoader(file_path)
            docs = loader.load()  # Load the document into a list of Document objects

        # Text Splitting - Break down large documents into smaller chunks for better retrieval
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
        splits = text_splitter.split_documents(docs)  # Split documents into chunks

        # Embeddings and Vector Store - Convert text to vectors and store in a vector database
        model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Pre-trained model for text embeddings
        st.session_state.vectorstore = Chroma.from_documents(
            documents=splits, embedding=HuggingFaceEmbeddings(model_name=model_name)
        )  # Create a vector store from document chunks

        # Create a retriever to fetch relevant documents based on similarity search
        st.session_state.retriever = st.session_state.vectorstore.as_retriever(
            search_kwargs={"k": 5}  # Retrieve top 5 most similar documents
        )

        # Create appropriate prompt based on source to specialize the LLM's responses
        # Use a generic system prompt that works for any knowledge source
        system_prompt = f"You are an expert in {source_name}. Answer questions based strictly on the provided context. Include the source metadata in your response."

        # Create RAG (Retrieval-Augmented Generation) chain for answering questions
        prompt_template = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    system_prompt
                ),  # System-level instruction
                HumanMessagePromptTemplate.from_template("""  # User prompt template
                Context:
                {context}

                Question:
                {question}
                """),
            ]
        )
        llm = ChatGroq(model="openai/gpt-oss-120b")  # Initialize the language model

        # Create the RAG chain that combines retrieval and generation
        st.session_state.rag_chain = (
            {
                "context": st.session_state.retriever,
                "question": RunnablePassthrough(),
            }  # Input dictionary
            | prompt_template  # Format the prompt with context and question
            | llm  # Generate response using the language model
            | StrOutputParser()  # Parse the output as a string
        )

        # Create Quiz Question Generation Chain to generate questions from the knowledge base
        interview_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    f"You are an expert in {source_name}. Based on the provided context, generate a challenging but fair 1 question that tests understanding of the material. Also provide a comprehensive reference answer. Format your response as:\n\nQUESTION: [Your question here]\nANSWER: [Your reference answer here]"
                ),  # System instruction for question generation
                HumanMessagePromptTemplate.from_template("""  # User prompt with context
                Context:
                {context}
                
                Generate a question and answer based on this context.
                """),
            ]
        )

        # Create the quiz generation chain that retrieves context and generates questions
        st.session_state.Interview_chain = (
            {
                "context": st.session_state.retriever
            }  # Input with just the context (no question)
            | interview_prompt  # Format the prompt with context
            | llm  # Generate question and answer using the language model
            | StrOutputParser()  # Parse the output as a string
        )

        # Create Answer Evaluation Chain to assess user responses
        evaluation_prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    f"You are an expert in {source_name}. Evaluate the user's answer compared to the reference answer. Provide a score from 0-10 and constructive feedback. Format your response as:\n\nSCORE: [0-10]\nFEEDBACK: [Your detailed feedback here]"
                ),  # System instruction for evaluation
                HumanMessagePromptTemplate.from_template("""  # User prompt with answers
                Reference Answer:
                {reference_answer}
                
                User Answer:
                {user_answer}
                
                Evaluate the user's answer.
                """),
            ]
        )

        # Create the evaluation chain that compares user answers to reference answers
        st.session_state.evaluation_chain = (
            {
                "reference_answer": RunnablePassthrough(),
                "user_answer": RunnablePassthrough(),
            }  # Input with both answers
            | evaluation_prompt  # Format the prompt with both answers
            | llm  # Generate evaluation using the language model
            | StrOutputParser()  # Parse the output as a string
        )

        st.session_state.current_source = (
            source_name  # Update the current source in session state
        )
        st.success(
            f"{source_name} knowledge base initialized successfully!"
        )  # Show success message
        return True  # Return True to indicate success
    except Exception as e:
        st.error(
            f"Error loading {source_name} knowledge base: {str(e)}"
        )  # Show error message
        return False  # Return False to indicate failure


# Create source selection buttons dynamically based on available files
# Get list of all pdf files in the data directory
import glob

data_files = glob.glob("data/*.pdf")
source_names = [os.path.basename(file).replace(".pdf", "") for file in data_files]

# Create columns for buttons (3 per row)
cols_per_row = 3
num_rows = (len(source_names) + cols_per_row - 1) // cols_per_row

for row in range(num_rows):
    cols = st.columns(cols_per_row)
    start_idx = row * cols_per_row
    end_idx = min(start_idx + cols_per_row, len(source_names))

    for i, source_name in enumerate(source_names[start_idx:end_idx]):
        with cols[i]:
            if st.button(source_name):  # Button to load knowledge base with file name
                load_knowledge_source(source_name)

# Display current knowledge source if one is selected
if st.session_state.current_source:
    st.info(
        f"Current knowledge source: {st.session_state.current_source}"
    )  # Show info message with current source

# Quiz Mode - The main functionality of the application
st.session_state.quiz_mode = True  # Set quiz mode to active

st.info('Click "Generate New Question" to start the quiz.')

# Generate new question button - Creates a single concise interview question
if st.session_state.current_source:
    if st.button(
        "Generate New Question"
    ):  # Button to generate a single interview question
        with st.spinner("Generating a question..."):  # Show loading spinner
            # Create a prompt for generating a single concise interview question
            interview_question_prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessagePromptTemplate.from_template(
                        f"You are an expert interviewer in {st.session_state.current_source}. Generate exactly ONE concise interview question that would be asked in a real job interview for a {st.session_state.current_source} position. Make it brief (under 20 words), practical, and focused on core knowledge. Just provide the question without any additional text or explanation."
                    ),
                    HumanMessagePromptTemplate.from_template(
                        "Generate a brief interview question."
                    ),
                ]
            )

            # Create a temporary chain for generating the interview question
            llm = ChatGroq(model="openai/gpt-oss-120b")
            interview_question_chain = (
                interview_question_prompt | llm | StrOutputParser()
            )

            # Generate the interview question
            interview_question = interview_question_chain.invoke({})

            # Store the question in session state for evaluation
            st.session_state.current_question = interview_question.strip()
            st.session_state.reference_answer = (
                "Reference answer will be generated after you submit your response."
            )

# Display current question section
if st.session_state.current_question:  # Check if a question is available
    st.write("### Question:")  # Section header
    st.write(st.session_state.current_question)  # Display the question

    # User answer input - Text area for user to enter their answer
    user_answer = st.text_area("Your answer:")  # Create a text input area

    # Evaluate answer button - Submits user answer for evaluation
    if st.button("Submit Answer") and user_answer and st.session_state.current_source:
        with st.spinner("Evaluating your answer..."):
            # Generate a simple reference answer if needed
            if (
                st.session_state.reference_answer
                == "Reference answer will be generated after you submit your response."
            ):
                llm = ChatGroq(model="openai/gpt-oss-120b")
                prompt = f"Short answer to: {st.session_state.current_question}"
                st.session_state.reference_answer = llm.invoke(prompt).content

            # Simple evaluation
            llm = ChatGroq(model="openai/gpt-oss-120b")
            eval_prompt = f"Rate 0-10: {user_answer} for question: {st.session_state.current_question}"
            evaluation = llm.invoke(eval_prompt).content

            # Parse evaluation response
            if "SCORE:" in evaluation and "FEEDBACK:" in evaluation:
                parts = evaluation.split("FEEDBACK:")
                score = parts[0].replace("SCORE:", "").strip()
                feedback = parts[1].strip()

                # Display evaluation
                st.write("### Evaluation:")
                st.write(f"SCORE -> {score}")
                st.write("(Score is out of 10)")
                st.write(feedback)

                # Show reference answer
                with st.expander("View Reference Answer"):
                    st.write(st.session_state.reference_answer)
            else:
                st.write("### Evaluation:")
                st.write(evaluation)

                # Show reference answer
                with st.expander("View Reference Answer"):
                    st.write(st.session_state.reference_answer)
