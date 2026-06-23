
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from main import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    POPULARITY_THRESHOLD,
    ROOT_DIR,
    load_dataset,
)

MODEL_PATH = ROOT_DIR / "models" / "rede_neural_mlp.pkl"


def main() -> None:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Modelo não encontrado. Execute primeiro: python src/main.py")

    model = joblib.load(MODEL_PATH)
    df = load_dataset()
    popular = df[df["popularity"] >= POPULARITY_THRESHOLD].sample(1, random_state=42)
    non_popular = df[df["popularity"] < POPULARITY_THRESHOLD].sample(2, random_state=42)
    samples = pd.concat([popular, non_popular], ignore_index=True)
    features = samples[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]

    for (_, row), prediction, probability in zip(samples.iterrows(), predictions, probabilities):
        label = "Popular" if prediction == 1 else "Não popular"
        actual = "Popular" if row["popularity"] >= POPULARITY_THRESHOLD else "Não popular"
        print("-" * 70)
        print(f"Faixa: {row['track_name']} — {row['artists']}")
        print(f"Gênero: {row['track_genre']} | Popularidade original: {row['popularity']}")
        print(f"Classe real: {actual}")
        print(f"Predição: {label}")
        print(f"Probabilidade de popularidade alta: {probability:.2%}")


if __name__ == "__main__":
    main()
