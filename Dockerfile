FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Environment variables for PYTHONPATH so src is discoverable
ENV PYTHONPATH=/app

# Copy requirements and install
COPY deploy_requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r deploy_requirements.txt

# Copy project files
COPY src/deployment.py src/
COPY src/preprocessing.py src/
COPY src/feature_engineering.py src/
COPY src/config.py src/
COPY src/aws_database.py src/
COPY model/best_model.pkl model/
COPY model/transformer.pkl model/

# Expose port for FastAPI
EXPOSE 8501

# Environment variables for model and transformer paths
ENV MODEL_PATH=/app/model/best_model.pkl
ENV TRANSFORMER_PATH=/app/model/transformer.pkl

# Default command: run Streamlit WebApp
CMD ["streamlit", "run", "src/deployment.py", "--server.port=8501", "--server.address=0.0.0.0"]