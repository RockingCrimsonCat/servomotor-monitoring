"""
Модуль 5: Реєстр користувачів і моторів.

Зберігає СПІЛЬНИЙ стан між усіма Streamlit-сесіями того самого
процесу. Реалізовано як singleton, доступ через `get_registry()`,
що внутрішньо використовує `@st.cache_resource` (Streamlit гарантує
один екземпляр на сервер).

Ролі:
- "operator" — моніторить власні мотори;
- "admin"    — бачить навантаження від усіх операторів і
               перенавчає моделі.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

try:
    import psutil
    _PSUTIL_OK = True
except ImportError:
    _PSUTIL_OK = False

from module1_data_generator import MotorPhysics, init_stream_state


WINDOW_SIZE = 200


@dataclass
class MotorState:
    """Стан одного сервомотора в реєстрі.

    Кожен мотор має ВЛАСНИЙ фізичний профіль (`physics`) з невеликими
    відхиленнями від номіналу — тому два мотори того самого оператора
    в одному режимі дадуть візуально РІЗНУ телеметрію (різні базові
    температури, рівні шуму, амплітуди тощо).

    Поле `description` — вільний текстовий опис локації / позначення
    обладнання, який задає оператор при створенні мотора
    (наприклад, "Станція 1 / Верстат 2", "Конвеєрна лінія A — двигун 3").
    """
    name: str
    owner: str
    physics: MotorPhysics = field(default_factory=MotorPhysics.default)
    description: str = ""
    mode: str = "NORMAL"
    connected: bool = True
    points_per_tick: int = 3
    rec_hold_sec: int = 5
    stream_state: dict = field(default=None)
    buffer: deque = field(default_factory=lambda: deque(maxlen=WINDOW_SIZE))
    tick_count: int = 0
    last_anomalies: int = 0
    last_state: str = "NORMAL"
    last_failure_prob: float = 0.0
    last_update_at: float = field(default_factory=time.time)
    created_at: float = field(default_factory=time.time)

    def __post_init__(self):
        if self.stream_state is None:
            self.stream_state = init_stream_state(seed=None, physics=self.physics)


@dataclass
class UserData:
    """Сесія користувача (живе доки працює сервер)."""
    name: str
    role: str
    motors: dict[str, MotorState] = field(default_factory=dict)
    login_at: float = field(default_factory=time.time)


class SystemRegistry:
    """Спільний реєстр усіх користувачів процесу.

    Модель доступу:
    - Адміністратор у системі ЄДИНИЙ: ім'я `ADMIN_NAME` ('admin'),
      воно зарезервоване й не може використовуватися операторами.
    - Оператори додаються адміністратором у `registered_operators`.
    - При спробі входу під незареєстрованим іменем — відмова.

    Безпечний для конкурентного доступу з кількох Streamlit-сесій
    завдяки внутрішньому `threading.Lock`.
    """

    ADMIN_NAME = "admin"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.users: dict[str, UserData] = {}
        self.registered_operators: set[str] = set()
        self._audit_log: list[dict] = []
        self.AUDIT_LIMIT = 200
        self.system_metrics: deque = deque(maxlen=200)
        self._cpu_primed = False

    def _log_event(self, actor: str, action: str, target: str = "",
                   details: str = "") -> None:
        """Внутрішнє: додати запис до аудит-логу.
        Викликати ПОЗА секцією, що тримає _lock, або обережно
        (зараз ми тримаємо лок під час викликів — додаємо без
        рекурсивного захоплення).
        """
        self._audit_log.append({
            "timestamp": time.time(),
            "actor": actor,
            "action": action,
            "target": target,
            "details": details,
        })
        if len(self._audit_log) > self.AUDIT_LIMIT:
            self._audit_log = self._audit_log[-self.AUDIT_LIMIT:]

    def get_audit_log(self, limit: int = 30) -> pd.DataFrame:
        """Останні `limit` подій у вигляді DataFrame для GUI."""
        with self._lock:
            events = list(self._audit_log[-limit:][::-1])
        if not events:
            return pd.DataFrame(
                columns=["Час", "Хто", "Дія", "Об'єкт", "Деталі"]
            )
        rows = []
        for e in events:
            rows.append({
                "Час": time.strftime("%H:%M:%S",
                                     time.localtime(e["timestamp"])),
                "Хто": e["actor"],
                "Дія": e["action"],
                "Об'єкт": e["target"],
                "Деталі": e["details"],
            })
        return pd.DataFrame(rows)

    def is_admin(self, name: str) -> bool:
        return name == self.ADMIN_NAME

    def can_login(self, name: str) -> tuple[bool, str]:
        """Перевірити, чи дозволено увійти під цим іменем.

        Повертає (allowed, role) — role є 'admin' або 'operator'.
        """
        with self._lock:
            if name == self.ADMIN_NAME:
                return True, "admin"
            if name in self.registered_operators:
                return True, "operator"
            return False, ""

    def register_user(self, name: str, role: str) -> UserData:
        """Створити запис активної сесії після успішної перевірки."""
        with self._lock:
            is_new = name not in self.users
            if is_new:
                self.users[name] = UserData(name=name, role=role)
            else:
                self.users[name].role = role
                self.users[name].login_at = time.time()
            self._log_event(actor=name, action="LOGIN", target=name,
                            details=f"role={role}, new={is_new}")
            return self.users[name]

    def logout_user(self, name: str) -> None:
        """М'який вихід: дані користувача залишаються в реєстрі."""
        with self._lock:
            self._log_event(actor=name, action="LOGOUT", target=name)

    def create_operator(self, name: str, by_admin: str = "admin") -> None:
        """Додати нового оператора (виконує адмін)."""
        with self._lock:
            if name == self.ADMIN_NAME:
                raise ValueError(
                    f"Ім'я '{self.ADMIN_NAME}' зарезервоване."
                )
            if not name:
                raise ValueError("Ім'я не може бути порожнім.")
            self.registered_operators.add(name)
            self._log_event(actor=by_admin, action="CREATE_OPERATOR",
                            target=name)

    def remove_operator(self, name: str, by_admin: str = "admin") -> None:
        """Видалити оператора повністю: з білого списку та з активних
        сесій (разом із його моторами).
        """
        with self._lock:
            self.registered_operators.discard(name)
            user = self.users.pop(name, None)
            details = ""
            if user is not None:
                details = f"motors_removed={len(user.motors)}"
            self._log_event(actor=by_admin, action="REMOVE_OPERATOR",
                            target=name, details=details)

    def list_registered_operators(self) -> list[str]:
        """Перелік ВСІХ зареєстрованих імен — навіть тих, хто зараз не онлайн."""
        with self._lock:
            return sorted(self.registered_operators)

    def list_operators(self) -> list[str]:
        """Перелік операторів, що мають активну сесію (онлайн)."""
        with self._lock:
            return sorted(
                n for n, u in self.users.items() if u.role == "operator"
            )

    def add_motor(self, owner: str, motor_name: str,
                  description: str = "",
                  physics: MotorPhysics | None = None) -> MotorState:
        """Додає новий мотор з ВИПАДКОВИМ фізичним профілем (±5–10%).

        Якщо `physics` не задано — генерується випадковий варіант,
        тому кожен новий мотор має унікальні характеристики.

        `description` — вільний опис локації обладнання, що задає
        оператор при створенні (наприклад, 'Станція 1 / Верстат 2').
        """
        with self._lock:
            user = self.users[owner]
            is_new = motor_name not in user.motors
            if is_new:
                if physics is None:
                    physics = MotorPhysics.random_variant(
                        np.random.default_rng(),
                        label=f"{owner}-{motor_name}",
                    )
                user.motors[motor_name] = MotorState(
                    name=motor_name, owner=owner,
                    physics=physics, description=description.strip(),
                )
                self._log_event(
                    actor=owner, action="ADD_MOTOR",
                    target=motor_name,
                    details=(f"description='{description.strip()}'"
                             if description.strip() else ""),
                )
            return user.motors[motor_name]

    def update_motor_description(self, owner: str, motor_name: str,
                                 description: str) -> None:
        """Оновити опис існуючого мотора."""
        with self._lock:
            user = self.users.get(owner)
            if user and motor_name in user.motors:
                user.motors[motor_name].description = description.strip()

    def remove_motor(self, owner: str, motor_name: str) -> None:
        with self._lock:
            removed = self.users[owner].motors.pop(motor_name, None)
            if removed is not None:
                self._log_event(actor=owner, action="REMOVE_MOTOR",
                                target=motor_name)

    def list_motors(self, owner: str) -> list[str]:
        with self._lock:
            user = self.users.get(owner)
            return sorted(user.motors.keys()) if user else []

    def get_motor(self, owner: str, motor_name: str) -> MotorState | None:
        with self._lock:
            user = self.users.get(owner)
            if not user:
                return None
            return user.motors.get(motor_name)

    def get_load_summary(self) -> pd.DataFrame:
        """Зведена таблиця: рядок на КОЖНОГО зареєстрованого оператора,
        включно з тими, хто зараз не онлайн (порожні поля).
        """
        with self._lock:
            now = time.time()
            rows = []
            for op_name in sorted(self.registered_operators):
                user = self.users.get(op_name)
                if user is None:
                    rows.append({
                        "Оператор": op_name,
                        "Статус": "OFFLINE",
                        "Моторів": 0,
                        "Активних": 0,
                        "У стані WEAR": 0,
                        "У стані FAILURE": 0,
                        "Тіків (всього)": 0,
                        "Аномалій у вікнах": 0,
                        "Тіків/с": 0.0,
                        "Сесія, с": 0,
                    })
                    continue
                motors = user.motors.values()
                total_ticks = sum(m.tick_count for m in motors)
                total_anom = sum(m.last_anomalies for m in motors)
                active = sum(1 for m in motors if m.connected)
                in_failure = sum(1 for m in motors if m.last_state == "FAILURE")
                in_wear = sum(1 for m in motors if m.last_state == "WEAR")
                session_age = max(1.0, now - user.login_at)
                ticks_per_sec = total_ticks / session_age
                rows.append({
                    "Оператор": op_name,
                    "Статус": "ONLINE",
                    "Моторів": len(user.motors),
                    "Активних": active,
                    "У стані WEAR": in_wear,
                    "У стані FAILURE": in_failure,
                    "Тіків (всього)": total_ticks,
                    "Аномалій у вікнах": total_anom,
                    "Тіків/с": round(ticks_per_sec, 2),
                    "Сесія, с": round(now - user.login_at, 0),
                })
            return pd.DataFrame(rows)

    def get_failing_motors(self) -> list[tuple[str, str, float]]:
        """Список (operator, motor, P_fail) для моторів у стані FAILURE."""
        with self._lock:
            result = []
            for username, user in self.users.items():
                if user.role != "operator":
                    continue
                for m in user.motors.values():
                    if m.last_state == "FAILURE":
                        result.append((username, m.name, m.last_failure_prob))
            return sorted(result, key=lambda x: -x[2])

    def sample_system_metrics(self) -> None:
        """Зафіксувати поточне навантаження CPU та RAM **нашого процесу**
        (Streamlit-сервер з ML-моделями, генерацією телеметрії тощо).

        - CPU: `Process.cpu_percent(interval=0.1)` блокуюче вимірювання
          протягом 100 мс — дає чесні цифри без шумових стрибків
          0↔100, які властиві виклику `interval=None`.
          Нормалізується на кількість логічних CPU, щоб шкала 0–100%
          відповідала "%-ту від усіх ядер" (а не одного).
        - RAM: `Process.memory_info().rss` — Resident Set Size (фізична
          пам'ять, реально зайнята процесом).
          Показуємо у МБ + % від системних.

        Викликається з адмін-дашборда на кожному auto-refresh.
        Зберігає до 200 останніх точок (≈ 5 хв при оновленні 1.5 с).
        Якщо `psutil` не встановлено — мовчки пропускає сэмплування.
        """
        if not _PSUTIL_OK:
            return
        try:
            proc = psutil.Process()
        except Exception:
            return

        if not self._cpu_primed:
            proc.cpu_percent(interval=None)
            self._cpu_primed = True
            return

        cpu_count = psutil.cpu_count(logical=True) or 1
        cpu_raw = float(proc.cpu_percent(interval=0.1))
        cpu_pct = cpu_raw / cpu_count

        mem_info = proc.memory_info()
        proc_rss_mb = float(mem_info.rss) / (1024 * 1024)
        vm = psutil.virtual_memory()
        sys_total_mb = float(vm.total) / (1024 * 1024)
        proc_mem_pct = (float(mem_info.rss) / float(vm.total)) * 100.0

        with self._lock:
            self.system_metrics.append({
                "timestamp": time.time(),
                "cpu_pct": cpu_pct,
                "mem_pct": proc_mem_pct,
                "mem_used_mb": proc_rss_mb,
                "mem_total_mb": sys_total_mb,
            })

    def get_system_metrics_df(self) -> pd.DataFrame:
        """DataFrame системних метрик з часом у форматі HH:MM:SS."""
        with self._lock:
            snapshot = list(self.system_metrics)
        if not snapshot:
            return pd.DataFrame(columns=["time", "cpu_pct", "mem_pct",
                                         "mem_used_mb", "mem_total_mb"])
        df = pd.DataFrame(snapshot)
        df["time"] = df["timestamp"].apply(
            lambda t: time.strftime("%H:%M:%S", time.localtime(t))
        )
        return df

    def latest_system_metric(self) -> dict | None:
        """Останнє зафіксоване значення CPU/RAM, або None."""
        with self._lock:
            if not self.system_metrics:
                return None
            return dict(self.system_metrics[-1])

    def system_totals(self) -> dict:
        with self._lock:
            operators = [u for u in self.users.values() if u.role == "operator"]
            total_motors = sum(len(u.motors) for u in operators)
            active_motors = sum(
                1 for u in operators for m in u.motors.values() if m.connected
            )
            total_ticks = sum(
                m.tick_count for u in operators for m in u.motors.values()
            )
            return {
                "operators": len(operators),
                "total_motors": total_motors,
                "active_motors": active_motors,
                "total_ticks": total_ticks,
            }


_singleton: SystemRegistry | None = None


def get_registry() -> SystemRegistry:
    """Повертає глобальний реєстр процесу.

    Використовуємо власний singleton (а не `@st.cache_resource`),
    щоб реєстр був доступним і у модулях, що імпортуються поза
    Streamlit-контекстом (тести, CLI-перевірки).
    """
    global _singleton
    if _singleton is None:
        _singleton = SystemRegistry()
    return _singleton
