"""
Модуль 2: Моделі машинного навчання.

Random Forest — класифікація стану сервомотора (NORMAL/WEAR/FAILURE).
Isolation Forest — виявлення аномальних точок у телеметрії.
Моделі тренуються один раз і зберігаються у файли .joblib.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

from module1_data_generator import TRAINING_CSV, save_training_dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

CLASSIFIER_PATH = os.path.join(MODEL_DIR, "rf_classifier.joblib")
ANOMALY_PATH = os.path.join(MODEL_DIR, "iforest.joblib")

FEATURES = ["temperature", "current", "speed", "vibration"]
CLASS_LABELS = ["NORMAL", "WEAR", "FAILURE"]


@dataclass
class TrainingReport:
    accuracy: float
    report_text: str


def _load_or_generate() -> pd.DataFrame:
    if not os.path.exists(TRAINING_CSV):
        save_training_dataset()
    return pd.read_csv(TRAINING_CSV)


def train_models(seed: int = 42) -> TrainingReport:
    df = _load_or_generate()
    X = np.asarray(df[FEATURES].to_numpy(), dtype=np.float64)
    y = np.asarray(df["state"].to_numpy(), dtype=object)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced",
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    report = classification_report(y_test, y_pred, digits=3)

    normal_mask = df["state"] == "NORMAL"
    iforest = IsolationForest(
        n_estimators=300,
        contamination=0.08,
        random_state=seed,
        n_jobs=-1,
    )
    iforest.fit(
        np.asarray(df.loc[normal_mask, FEATURES].to_numpy(), dtype=np.float64)
    )

    joblib.dump(clf, CLASSIFIER_PATH)
    joblib.dump(iforest, ANOMALY_PATH)

    return TrainingReport(accuracy=accuracy, report_text=report)


def models_exist() -> bool:
    return os.path.exists(CLASSIFIER_PATH) and os.path.exists(ANOMALY_PATH)


def load_models() -> tuple[RandomForestClassifier, IsolationForest]:
    if not models_exist():
        train_models()
    clf = joblib.load(CLASSIFIER_PATH)
    iforest = joblib.load(ANOMALY_PATH)
    return clf, iforest


def predict_state(clf: RandomForestClassifier,
                  samples: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(samples[FEATURES].to_numpy(), dtype=np.float64)
    labels = clf.predict(X)
    proba = clf.predict_proba(X)
    return labels, proba


def detect_anomalies(iforest: IsolationForest,
                     samples: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(samples[FEATURES].to_numpy(), dtype=np.float64)
    raw = iforest.predict(X)
    score = iforest.decision_function(X)
    return raw == -1, score


def class_index(clf: RandomForestClassifier, label: str) -> int:
    classes = list(clf.classes_)
    if label not in classes:
        raise ValueError(f"Клас {label} відсутній у моделі: {classes}")
    return classes.index(label)


if __name__ == "__main__":
    report = train_models()
    print(f"[Модуль 2] Random Forest accuracy: {report.accuracy:.3f}")
    print(report.report_text)
    print(f"Моделі збережено у: {MODEL_DIR}")
