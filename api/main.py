from fastapi import FastAPI # type: ignore
from pydantic import BaseModel # type: ignore
import psycopg2, os
import pandas as pd
from joblib import load

app = FastAPI(title="Fraud Detection API")

# Подключение к базе данных Postgres
DB = os.getenv("DATABASE_URL", "postgresql://app:app@frauddb:5432/frauddb")

# Загружаем обученную модель
model_path = "/app/models/rf_model.joblib"
try:
    bundle = load(model_path)
    model = bundle["model"]
    scaler = bundle.get("scaler", None)
    columns = bundle["columns"]
    print("✅ Model loaded successfully!")
except Exception as e:
    print("⚠️ Failed to load model:", e)
    model, scaler, columns = None, None, []

# Проверка, что API работает
@app.get("/")
def home():
    return {"message": "Hello from Fraud API"}

@app.get("/health")
def health():
    try:
        psycopg2.connect(DB).close()
        return {"status": "ok", "db": "connected"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

# --- Новая часть: предсказание фрода ---

class Transaction(BaseModel):
    Time: float
    Amount: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float

@app.post("/predict")
def predict(transaction: Transaction):
    if model is None:
        return {"error": "Model not loaded"}

    # Преобразуем входные данные в DataFrame
    df = pd.DataFrame([transaction.dict()])
    if "Amount" in df.columns:
        df["scaled_amount"] = (df["Amount"] - df["Amount"].mean()) / df["Amount"].std()
    if "Time" in df.columns:
        df["scaled_time"] = (df["Time"] - df["Time"].mean()) / df["Time"].std()
    df = df.drop(columns=["Amount", "Time"], errors="ignore")

    # Убедимся, что порядок колонок совпадает с обучением
    for col in columns:
        if col not in df.columns:
            df[col] = 0
    df = df[columns]

    proba = model.predict_proba(df)[0, 1]
    is_fraud = bool(proba >= 0.5)

    return {
        "fraud_probability": round(float(proba), 4),
        "is_fraud": is_fraud
    }

@app.get("/frauds")
def get_fraud_examples(
    limit: int = 10,
    amount_min: float = 0.0,
    time_min: float = 0.0,
    sort: str = "amount_desc"
):
    
    import pandas as pd
    import os

    data_path = "/app/data/creditcard.csv"
    if not os.path.exists(data_path):
        return {"error": "Dataset not found inside container"}

    df = pd.read_csv(data_path)

    # Фильтруем только мошеннические операции
    frauds = df[df["Class"] == 1]

    # Применяем фильтры
    frauds = frauds[frauds["Amount"] >= amount_min]
    frauds = frauds[frauds["Time"] >= time_min]

    # Сортировка
    if sort == "amount_desc":
        frauds = frauds.sort_values(by="Amount", ascending=False)
    elif sort == "amount_asc":
        frauds = frauds.sort_values(by="Amount", ascending=True)
    elif sort == "time_desc":
        frauds = frauds.sort_values(by="Time", ascending=False)
    elif sort == "time_asc":
        frauds = frauds.sort_values(by="Time", ascending=True)

    # Возвращаем только нужное количество строк
    return frauds.head(limit).to_dict(orient="records")
