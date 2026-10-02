FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY infer ./infer
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -e .
EXPOSE 8000
CMD ["python", "-m", "infer.server", "--port", "8000"]
