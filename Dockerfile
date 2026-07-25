FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[phonetics]"

COPY data ./data
COPY reviewer_app.py ./
RUN medterm-build

EXPOSE 8000 8501
CMD ["medterm-api", "--host", "0.0.0.0", "--port", "8000"]
