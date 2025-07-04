FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY train.py .
COPY deliverytime.csv .

RUN mkdir -p models

CMD ["python", "train.py"]
