import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from joblib import dump
from pathlib import Path
import os, urllib.request, zipfile

# 1️⃣ Проверяем, есть ли датасет локально
data_path = Path("data/creditcard.csv")

if not data_path.exists():
    print("📥 Dataset not found locally, downloading...")
    # Прямая ссылка на CSV (копия с Kaggle)
    url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
    Path("data").mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, data_path)
    print("✅ Downloaded creditcard.csv")

# 2️⃣ Загружаем датасет
df = pd.read_csv(data_path)
print("✅ Dataset loaded successfully!")
print(df.head())
print(f"Total samples: {len(df)}")


# 3️⃣ Подготовка признаков
X = df.drop(columns=["Class"])
y = df["Class"]

# Масштабируем Amount и Time
scaler = StandardScaler()
X["scaled_amount"] = scaler.fit_transform(X["Amount"].values.reshape(-1, 1))
X["scaled_time"] = scaler.fit_transform(X["Time"].values.reshape(-1, 1))
X = X.drop(columns=["Amount", "Time"])

# 4️⃣ Делим на обучающую и тестовую выборку
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5️⃣ Обучаем модель RandomForest
print("\n🚀 Training RandomForest model...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)
model.fit(X_train, y_train)

# 6️⃣ Оцениваем качество
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n📊 Evaluation metrics:")
print(classification_report(y_test, y_pred, digits=4))
print("ROC-AUC:", roc_auc_score(y_test, y_proba))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 7️⃣ Сохраняем модель
Path("models").mkdir(parents=True, exist_ok=True)
dump({"model": model, "scaler": scaler, "columns": list(X.columns)}, "models/rf_model.joblib")
print("\n💾 Model saved to models/rf_model.joblib")
