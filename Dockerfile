
# Use Python 3.9 image as base
FROM python:3.9

# Set the working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y     gcc     g++     && rm -rf /var/lib/apt/lists/*

# Copy requirements file first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir sentence-transformers

# Copy the application code
COPY . .

# Create the data directory if it doesn't exist
RUN mkdir -p data

# Expose the port the app runs on (Hugging Face Spaces uses port 7860)
EXPOSE 7860

# Command to run the application on Hugging Face Spaces
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
