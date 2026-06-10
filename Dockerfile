FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

COPY ml /app/ml
COPY data /app/data
COPY models /app/models

CMD ["python", "/app/ml/train_model.py"]
