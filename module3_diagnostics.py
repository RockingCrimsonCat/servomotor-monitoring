"""
Модуль 3: Діагностика та прогноз стану сервомотора.

Формує:
- кольоровий індикатор стану (green/yellow/red);
- ймовірність відмови у відсотках;
- рівень ризику (Low / Medium / High);
- набір текстових рекомендацій залежно від виявлених порушень.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from module2_ml_models import FEATURES, class_index


THRESHOLDS = {
    "temperature_warn": 55.0,
    "temperature_alarm": 70.0,
    "current_warn": 8.5,
    "current_alarm": 11.0,
    "speed_low": 3500.0,
    "vibration_warn": 2.5,
    "vibration_alarm": 4.5,
}

STATE_COLORS = {
    "NORMAL": "#2ecc71",
    "WEAR": "#f1c40f",
    "FAILURE": "#e74c3c",
}


@dataclass
class DiagnosticReport:
    state: str
    color: str
    failure_probability: float
    risk_level: str
    recommendations: list[str] = field(default_factory=list)
    class_probabilities: dict[str, float] = field(default_factory=dict)
    anomaly_rate: float = 0.0


def _risk_level(probability_percent: float) -> str:
    if probability_percent < 25.0:
        return "Low"
    if probability_percent < 60.0:
        return "Medium"
    return "High"


def _build_recommendations(latest: pd.Series, state: str,
                           anomaly_rate: float) -> list[str]:
    rec: list[str] = []
    t = latest.get("temperature", 0.0)
    i = latest.get("current", 0.0)
    s = latest.get("speed", 0.0)
    v = latest.get("vibration", 0.0)

    if t >= THRESHOLDS["temperature_alarm"]:
        rec.append(
            f"Критичний перегрів обмотки ({t:.1f} °C): негайно зупинити двигун "
            f"та перевірити систему охолодження."
        )
    elif t >= THRESHOLDS["temperature_warn"]:
        rec.append(
            f"Підвищена температура ({t:.1f} °C): зменшити навантаження або "
            f"тривалість безперервної роботи."
        )

    if i >= THRESHOLDS["current_alarm"]:
        rec.append(
            f"Аварійний струм ({i:.2f} A): можливе коротке замикання або "
            f"заклинювання вала, потрібна інспекція обмотки та підшипників."
        )
    elif i >= THRESHOLDS["current_warn"]:
        rec.append(
            f"Підвищене споживання струму ({i:.2f} A): перевірити навантаження "
            f"та якість контактів."
        )

    if v >= THRESHOLDS["vibration_alarm"]:
        rec.append(
            f"Сильна вібрація ({v:.2f} мм/с): ймовірний дисбаланс ротора або "
            f"знос підшипників, потрібна заміна вузла."
        )
    elif v >= THRESHOLDS["vibration_warn"]:
        rec.append(
            f"Підвищена вібрація ({v:.2f} мм/с): провести балансування ротора "
            f"та змащення підшипників."
        )

    if s <= THRESHOLDS["speed_low"]:
        rec.append(
            f"Знижена швидкість обертання ({s:.0f} об/хв): перевірити момент "
            f"опору навантаження та керування ШІМ."
        )

    if anomaly_rate >= 0.30:
        rec.append(
            f"Аномалії складають {anomaly_rate*100:.0f}% останніх вимірювань — "
            f"провести позачергову діагностику."
        )

    if state == "NORMAL" and not rec:
        rec.append("Параметри в межах норми. Продовжити штатну експлуатацію.")
    if state == "WEAR" and not rec:
        rec.append("Ознаки зносу: запланувати технічне обслуговування у найближчий цикл.")
    if state == "FAILURE" and not rec:
        rec.append("Стан відмови: зупинити обладнання та звернутися до сервісної служби.")

    return rec


def build_report(window_df: pd.DataFrame,
                 clf,
                 anomaly_mask: np.ndarray | None = None) -> DiagnosticReport:
    """Сформувати діагностичний звіт по останньому вікну вимірювань.

    Параметри
    ---------
    window_df : DataFrame з колонками FEATURES (останні N точок).
    clf       : натренований RandomForestClassifier.
    anomaly_mask : булева маска довжиною len(window_df) з аномаліями Isolation Forest.
    """
    if window_df.empty:
        return DiagnosticReport(
            state="NORMAL",
            color=STATE_COLORS["NORMAL"],
            failure_probability=0.0,
            risk_level="Low",
            recommendations=["Очікування телеметрії..."],
        )

    X = np.asarray(window_df[FEATURES].to_numpy(), dtype=np.float64)
    proba = clf.predict_proba(X)

    mean_proba = proba.mean(axis=0)
    classes = list(clf.classes_)
    class_probs = {cls: float(mean_proba[idx]) for idx, cls in enumerate(classes)}

    state = max(class_probs, key=class_probs.get)
    failure_idx = class_index(clf, "FAILURE")
    wear_idx = class_index(clf, "WEAR")
    failure_prob = (mean_proba[failure_idx] + 0.5 * mean_proba[wear_idx]) * 100.0
    failure_prob = float(np.clip(failure_prob, 0.0, 100.0))

    anomaly_rate = 0.0
    if anomaly_mask is not None and len(anomaly_mask) > 0:
        anomaly_rate = float(np.mean(anomaly_mask))

    recommendations = _build_recommendations(
        window_df.iloc[-1], state, anomaly_rate
    )

    return DiagnosticReport(
        state=state,
        color=STATE_COLORS[state],
        failure_probability=failure_prob,
        risk_level=_risk_level(failure_prob),
        recommendations=recommendations,
        class_probabilities=class_probs,
        anomaly_rate=anomaly_rate,
    )


if __name__ == "__main__":
    from module1_data_generator import generate_dataset
    from module2_ml_models import detect_anomalies, load_models

    clf, iforest = load_models()
    df = generate_dataset(samples_per_mode=50, seed=7).tail(50)
    mask, _ = detect_anomalies(iforest, df)
    rep = build_report(df, clf, mask)
    print(f"Стан: {rep.state} ({rep.color})")
    print(f"Імовірність відмови: {rep.failure_probability:.1f}%  ризик: {rep.risk_level}")
    print(f"Аномалії у вікні: {rep.anomaly_rate*100:.1f}%")
    for r in rep.recommendations:
        print(" -", r)
