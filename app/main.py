import pickle
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict

# Инициализируем FastAPI
app = FastAPI()

# разрешаем разным источникам использованить наш API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ReviewInput(BaseModel):
    review: str


# Загружаем векторизотор и модели из пикл файлов
with open("app/models/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

aspect_models = {}
for aspect in ["praktika", "teoriya", "prepodavatel", "tekhnologii", "aktualnost"]:
    with open(f"app/models/{aspect.lower()}_model.pkl", "rb") as f:
        aspect_models[aspect] = pickle.load(f)


# 1. Предобрабатываем текст
def preprocess_text(text: str) -> str:
    import re

    text = text.lower()
    text = re.sub(r"\d+", "", text)  # Удаляем цифры
    text = re.sub(r"[^\w\s]", "", text)  # Удаляем знаки препинания
    return text


# 2. Функция предсказывания
def predict_aspects(review: str) -> Dict[str, int]:
    review_processed = preprocess_text(review)
    review_tfidf = vectorizer.transform(
        [review_processed]
    )  # Конвертируем отзыв в TF-IDF

    predictions = {}
    for aspect, model in aspect_models.items():
        predictions[aspect] = int(
            model.predict(review_tfidf)[0]
        )  # Конвертируем numpy в Python int
    return predictions


# Объявляем маршрут для запросов ^^
@app.post("/predict")
async def predict_review_aspects(input_data: ReviewInput):
    review = input_data.review
    predictions = predict_aspects(review)
    return predictions
