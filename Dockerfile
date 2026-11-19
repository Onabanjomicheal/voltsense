# Battery SoC Estimator API
# Model + scaler are baked into the image at build time (pinned decision: simpler,
# reproducible, no runtime fetch dependency -- Render's free tier already has slow
# cold starts, adding a network fetch to every cold start would make it worse).

FROM python:3.11-slim

RUN apt-get update && apt-get upgrade -y --no-install-recommends && rm -rf /var/lib/apt/lists/*

WORKDIR /appE

COPY requirements-serving.txt .
RUN pip install --no-cache-dir --upgrade pip "wheel>=0.46.2" setuptools && pip install --no-cache-dir -r requirements-serving.txt && pip uninstall -y setuptools

COPY src/ ./src/

# Must exist before building. Run `dvc repro` or `python -m src.training.train
# --config configs/lstm.yaml` first if they don't.
COPY models/lstm_model.pth ./models/lstm_model.pth
COPY models/scaler.joblib ./models/scaler.joblib

ENV MODEL_PATH=/app/models/lstm_model.pth
ENV SCALER_PATH=/app/models/scaler.joblib

ENV PORT=8000
EXPOSE 8000

CMD uvicorn src.serving.api:app --host 0.0.0.0 --port ${PORT}
