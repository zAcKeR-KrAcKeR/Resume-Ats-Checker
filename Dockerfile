# Use an official lightweight Python image
FROM python:3.12-slim

# Set working directory in the container
WORKDIR /app

# Copy only requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy English model
RUN python -m spacy download en_core_web_sm

# Copy the rest of your application code
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Set the default command to run the Streamlit app
CMD ["streamlit", "run", "main.py"]
