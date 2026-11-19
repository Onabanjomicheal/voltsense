
# Battery SoC Estimator API
# Model + scaler are baked into the image at build time.

FROM python:3.11-slim

# Update operating system packages.
RUN apt-get update \
    && apt-get upgrade -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy serving dependencies.
COPY requirements-serving.txt .

# Install Python dependencies.
RUN pip install --no-cache-dir --upgrade pip "wheel>=0.46.2" \
    && pip install --no-cache-dir -r requirements-serving.txt \
    && pip install --no-cache-dir --upgrade "setuptools>=78.1.1"

# Verify setuptools is not vulnerable.
RUN python -c "import setuptools; v=tuple(map(int, setuptools.__version__.split('.')[:3])); print('Installed setuptools:', setuptools.__version__); assert v >= (78, 1, 1), 'Vulnerable setuptools version detected'"

# Verify dependency consistency.
RUN pip check

# Copy application source.
COPY src/ ./src/

# Bake trained model and scaler into the image.
COPY models/lstm_model.pth ./models/lstm_model.pth
COPY models/scaler.joblib ./models/scaler.joblib

# API configuration.
ENV MODEL_PATH=/app/models/lstm_model.pth
ENV SCALER_PATH=/app/models/scaler.joblib
ENV PORT=8000

EXPOSE 8000

CMD uvicorn src.serving.api:app --host 0.0.0.0 --port ${PORT}
