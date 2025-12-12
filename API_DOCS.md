# API Documentation

This document explains how to use the GET and POST endpoints of the Interview Chatbot API.

## Base URL
```
https://hema01-chatbot-simulation-interview.hf.space
```

## Endpoints

### 1. Root Endpoint
**GET** `/`
- Returns basic API information
- No parameters required
- Example response:
```json
{
  "message": "Interview Q&A API",
  "version": "1.0.0",
  "endpoints": {
    "api_info": "/api/info - Get API information",
    "sources": "/api/sources - Get list of available knowledge sources",
    "select_source": "/api/select_source - Select a knowledge source",
    "generate_question": "/api/generate_question - Generate an interview question",
    "submit_answer": "/api/submit_answer - Submit an answer for evaluation",
  }
}
```

### 2. Get Available Sources
**GET** `/api/sources`
- Returns a list of available knowledge sources
- No parameters required
- Example response:
```json
["Data Analyst", "Data Science", "Mechanical Engineering", "Planning Engineers", "SQL", "Software Engineering"]
```

### 3. Select Knowledge Source
**POST** `/api/select_source`
- Selects a knowledge source to use for Q&A
- Request body:
```json
{
  "source_name": "Data Science"
}
```
- Example response:
```json
{
  "success": true,
  "message": "Data Science knowledge base initialized successfully!"
}
```

### 4. Generate Interview Question
**GET** `/api/generate_question`
- Generates a new interview question from the selected knowledge source
- No parameters required (requires a source to be selected first)
- Example response:
```json
{
  "question": "What is the difference between supervised and unsupervised learning?",
  "success": true,
  "message": "Question generated successfully"
}
```

### 5. Submit Answer
**POST** `/api/submit_answer`
- Submits an answer to the current interview question and gets evaluation
- Request body:
```json
{
  "answer": "Supervised learning uses labeled data while unsupervised learning works with unlabeled data to find patterns."
}
```
- Example response:
```json
{
  "score": "8/10",
  "feedback": "Good answer, but could include examples of algorithms for each type.",
  "reference_answer": "Supervised learning uses labeled training data to learn input-output mappings, while unsupervised learning finds hidden patterns in unlabeled data.",
  "success": true,
  "message": "Answer evaluated successfully"
}
```

## Usage Workflow

1. Get available sources using `GET /api/sources`
2. Select a source using `POST /api/select_source`
3. Generate a question using `GET /api/generate_question`
4. Submit an answer using `POST /api/submit_answer`
5. Repeat steps 3-4 as needed

## Error Responses

All endpoints return appropriate HTTP status codes and error messages:

- 400: Bad Request (missing parameters, invalid data)
- 500: Internal Server Error (unexpected issues)

Example error response:
```json
{
  "detail": "No knowledge source selected. Please select a source first."
}
```
