---
title: Chatbot Interview Simulation
emoji: 🐠
colorFrom: yellow
colorTo: blue
sdk: docker
pinned: false
---

# Chatbot Interview Simulation

This is an interactive chatbot application that simulates job interviews across different professional fields. The application uses a Retrieval-Augmented Generation (RAG) pipeline to provide contextually relevant interview questions and feedback.

## Features

- Interview simulations for various professional fields:
  - Data Analyst
  - Data Science
  - Mechanical Engineering
  - Planning Engineers
  - SQL
  - Software Engineering
- Context-aware question generation
- Real-time feedback on answers
- User-friendly interface

## How it Works

The application uses a RAG (Retrieval-Augmented Generation) pipeline that:
1. Retrieves relevant information from domain-specific documents
2. Generates appropriate interview questions based on the retrieved context
3. Evaluates user responses and provides feedback

## Setup Instructions

1. Clone this repository to your local machine
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   python main.py
   ```

## Hugging Face Space Deployment

This application is set up to run on Hugging Face Spaces with the following configuration:
- SDK: Docker
- License: None
- Hardware: CPU Basic

## Project Structure

```
├── Dockerfile           # Docker configuration
├── data/               # Domain-specific documents
├── helpers/            # Utility functions and configurations
├── main.py             # Application entry point
├── requirements.txt    # Python dependencies
├── routes/             # API routes
└── services/           # Core business logic
```

## Usage

1. Select a professional field for your interview
2. Answer the questions asked by the chatbot
3. Receive feedback on your responses
4. Continue with follow-up questions or start a new interview

## Technologies Used

- Python
- RAG (Retrieval-Augmented Generation)
- Docker
- FastAPI (inferred from structure)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project does not have a specific license.
