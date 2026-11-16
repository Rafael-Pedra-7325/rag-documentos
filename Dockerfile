FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/app/data \
    HF_HOME=/app/data/huggingface

WORKDIR /app

# PyTorch só para CPU: a imagem fica bem menor que com a build CUDA padrão.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml README.md ./
COPY src ./src
COPY examples ./examples
RUN pip install ".[all]"

RUN useradd --create-home app && mkdir -p /app/data && chown -R app:app /app
USER app
EXPOSE 8000

CMD ["uvicorn", "rag_documentos.api:app", "--host", "0.0.0.0", "--port", "8000"]
