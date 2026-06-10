"""
Модуль 1: Генератор синтетичної телеметрії сервомотора.

Параметри: температура (°C), струм (A), швидкість (об/хв), вібрація (мм/с).
Режими: NORMAL, WEAR, FAILURE. Дані формуються за спрощеними фізичними
співвідношеннями електродвигуна постійного струму з накладеним шумом
і зберігаються у CSV для подальшого навчання моделей.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

TRAINING_CSV = os.path.join(DATA_DIR, "training_dataset.csv")
LIVE_CSV = os.path.join(DATA_DIR, "live_stream.csv")
STREAMS_DIR = os.path.join(DATA_DIR, "streams")
os.makedirs(STREAMS_DIR, exist_ok=True)

NOMINAL_VOLTAGE = 24.0
RESISTANCE_NORMAL = 0.6
TORQUE_CONSTANT = 0.045
BACK_EMF_CONSTANT = 0.0395
NOMINAL_LOAD_TORQUE = 0.25
AMBIENT_TEMP = 25.0
THERMAL_RESISTANCE = 0.42
RAD_PER_S_TO_RPM = 60.0 / (2.0 * np.pi)


@dataclass
class MotorPhysics:
    voltage: float = NOMINAL_VOLTAGE
    resistance: float = RESISTANCE_NORMAL
    torque_constant: float = TORQUE_CONSTANT
    back_emf_constant: float = BACK_EMF_CONSTANT
    load_torque: float = NOMINAL_LOAD_TORQUE
    ambient_temp: float = AMBIENT_TEMP
    thermal_resistance: float = THERMAL_RESISTANCE
    noise_scale: float = 1.0
    label: str = "standard"

    @classmethod
    def default(cls) -> "MotorPhysics":
        return cls()

    @classmethod
    def random_variant(cls, rng: np.random.Generator | None = None,
                       label: str = "factory-grade",
                       spread: float = 1.0) -> "MotorPhysics":
        if rng is None:
            rng = np.random.default_rng()
        return cls(
            voltage=NOMINAL_VOLTAGE,
            resistance=RESISTANCE_NORMAL * float(1 + rng.normal(0, 0.08 * spread)),
            torque_constant=TORQUE_CONSTANT * float(1 + rng.normal(0, 0.05 * spread)),
            back_emf_constant=BACK_EMF_CONSTANT * float(1 + rng.normal(0, 0.05 * spread)),
            load_torque=NOMINAL_LOAD_TORQUE * float(1 + rng.normal(0, 0.10 * spread)),
            ambient_temp=AMBIENT_TEMP + float(rng.uniform(-3 * spread, 5 * spread)),
            thermal_resistance=THERMAL_RESISTANCE * float(1 + rng.normal(0, 0.10 * spread)),
            noise_scale=float(1.0 + rng.uniform(-0.20 * spread, 0.30 * spread)),
            label=label,
        )

    def summary(self) -> str:
        return (
            f"V={self.voltage:.1f}V, R={self.resistance:.3f}Om, "
            f"k_t={self.torque_constant:.4f}, "
            f"T_amb={self.ambient_temp:.1f}C, "
            f"noise x{self.noise_scale:.2f}"
        )


DEFAULT_PHYSICS = MotorPhysics.default()


@dataclass
class ModeProfile:
    label: str
    friction_mult: float
    resistance_mult: float
    vib_base: float
    vib_noise: float
    temp_offset: float
    failure_spike_prob: float


MODES: dict[str, ModeProfile] = {
    "NORMAL": ModeProfile(
        label="NORMAL",
        friction_mult=1.00,
        resistance_mult=1.00,
        vib_base=0.8,
        vib_noise=0.15,
        temp_offset=0.0,
        failure_spike_prob=0.0,
    ),
    "WEAR": ModeProfile(
        label="WEAR",
        friction_mult=1.35,
        resistance_mult=1.15,
        vib_base=2.2,
        vib_noise=0.45,
        temp_offset=6.0,
        failure_spike_prob=0.01,
    ),
    "FAILURE": ModeProfile(
        label="FAILURE",
        friction_mult=1.90,
        resistance_mult=1.45,
        vib_base=5.5,
        vib_noise=1.20,
        temp_offset=18.0,
        failure_spike_prob=0.08,
    ),
}


def _sample_load_torque(rng: np.random.Generator, mode: ModeProfile,
                        physics: MotorPhysics) -> float:
    base = physics.load_torque * mode.friction_mult
    return float(base + rng.normal(0.0, 0.05 * physics.noise_scale))


def _physics_step(rng: np.random.Generator, mode: ModeProfile,
                  prev_temp: float,
                  physics: MotorPhysics = DEFAULT_PHYSICS) -> dict[str, float]:

    ns = physics.noise_scale
    load_torque = _sample_load_torque(rng, mode, physics)
    resistance = physics.resistance * mode.resistance_mult

    current = load_torque / physics.torque_constant + rng.normal(0.0, 0.25 * ns)
    current = max(current, 0.05)

    omega = (physics.voltage - current * resistance) / physics.back_emf_constant
    speed_rpm = omega * RAD_PER_S_TO_RPM + rng.normal(0.0, 25.0 * ns)
    speed_rpm = max(speed_rpm, 0.0)

    power_loss = current ** 2 * resistance
    target_temp = (physics.ambient_temp + power_loss * physics.thermal_resistance
                   + mode.temp_offset)
    temp = prev_temp + 0.08 * (target_temp - prev_temp) + rng.normal(0.0, 0.8 * ns)

    vibration = (mode.vib_base + abs(rng.normal(0.0, mode.vib_noise * ns))
                 + rng.normal(0.0, 0.2 * ns))
    vibration = max(vibration, 0.05)
    if rng.random() < mode.failure_spike_prob:
        vibration += rng.uniform(2.0, 5.0)
        temp += rng.uniform(1.0, 3.0)
        current += rng.uniform(0.2, 0.6)

    return {
        "temperature": float(temp),
        "current": float(current),
        "speed": float(speed_rpm),
        "vibration": float(vibration),
    }


def generate_dataset(samples_per_mode: int = 1500,
                     seed: int = 42,
                     num_profiles_per_mode: int = 30,
                     training_spread: float = 1.5) -> pd.DataFrame:

    rng = np.random.default_rng(seed)
    records: list[dict] = []
    samples_per_profile = max(1, samples_per_mode // num_profiles_per_mode)

    for mode_name, profile in MODES.items():
        for _ in range(num_profiles_per_mode):
            physics = MotorPhysics.random_variant(rng, spread=training_spread)
            temp = physics.ambient_temp + profile.temp_offset
            for _ in range(samples_per_profile):
                row = _physics_step(rng, profile, temp, physics)
                temp = row["temperature"]
                row["state"] = mode_name
                records.append(row)

    df = pd.DataFrame.from_records(records)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def save_training_dataset(samples_per_mode: int = 1500) -> str:
    df = generate_dataset(samples_per_mode=samples_per_mode)
    df.to_csv(TRAINING_CSV, index=False)
    return TRAINING_CSV


def stream_step(state: dict, mode_name: str = "NORMAL") -> dict[str, float]:
    profile = MODES[mode_name]
    physics = state.get("physics", DEFAULT_PHYSICS)
    row = _physics_step(state["rng"], profile, state["temp"], physics)
    state["temp"] = row["temperature"]
    row["state_true"] = mode_name
    row["timestamp"] = time.time()
    return row


def init_stream_state(seed: int | None = None,
                      physics: MotorPhysics | None = None) -> dict:
    p = physics if physics is not None else DEFAULT_PHYSICS
    return {
        "rng": np.random.default_rng(seed),
        "temp": p.ambient_temp,
        "physics": p,
    }


def _sanitize(name: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in name)


def stream_path(owner: str, motor_name: str) -> str:
    return os.path.join(
        STREAMS_DIR, f"{_sanitize(owner)}__{_sanitize(motor_name)}.csv"
    )


def append_live_row(row: dict, owner: str | None = None,
                    motor_name: str | None = None) -> None:
    df = pd.DataFrame([row])
    if owner and motor_name:
        path = stream_path(owner, motor_name)
    else:
        path = LIVE_CSV
    header = not os.path.exists(path)
    df.to_csv(path, mode="a", header=header, index=False)


if __name__ == "__main__":
    path = save_training_dataset()
    print(f"[Модуль 1] Навчальний датасет збережено: {path}")
    sample = pd.read_csv(path).head()
    print(sample.to_string(index=False))
