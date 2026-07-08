# 1. Use the exact python version matching your local environment
FROM python:3.10-slim

# 2. Set working directory
WORKDIR /app

# 3. Copy ONLY requirements first to leverage Docker layer caching
COPY requirements.txt .

# 4. Install dependencies (Cached unless requirements.txt changes)
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the workspace files
COPY src/ ./src/
COPY models/ ./models/
COPY IT_customer_churn.csv .

# 6. Expose FastAPI's default port
EXPOSE 8000

# 7. Run production server using uvicorn
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]