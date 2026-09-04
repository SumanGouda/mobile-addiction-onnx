FROM python:3.12-slim

WORKDIR /app

# Copy everything needed for the package to be installable
COPY pyproject.toml .
COPY src/ src/
COPY app/ app/
COPY utils/ utils/
COPY configs/ configs/

RUN pip install --upgrade pip && pip install ".[tracking]"

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
