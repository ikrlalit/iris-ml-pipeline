#Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY app/ ./app/
COPY app/model.py ./

RUN pip install --no-cache-dir -r app/requirements.txt

# Train model inside container during build
RUN python model.py

# Serve API
CMD ["uvicorn", "app.predict:app", "--host", "0.0.0.0", "--port", "80"]

