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
        with self._lock:
            if name == self.ADMIN_NAME:
                return True, "admin"
            if name in self.registered_operators:
                return True, "operator"
            return False, ""

    def register_user(self, name: str, role: str) -> UserData:
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
        with self._lock:
            self._log_event(actor=name, action="LOGOUT", target=name)

    def create_operator(self, name: str, by_admin: str = "admin") -> None:
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
        with self._lock:
            self.registered_operators.discard(name)
            user = self.users.pop(name, None)
            details = ""
            if user is not None:
                details = f"motors_removed={len(user.motors)}"
            self._log_event(actor=by_admin, action="REMOVE_OPERATOR",
                            target=name, details=details)

    def list_registered_operators(self) -> list[str]:
        with self._lock:
            return sorted(self.registered_operators)

    def list_operators(self) -> list[str]:
        with self._lock:
            return sorted(
                n for n, u in self.users.items() if u.role == "operator"
            )

    def add_motor(self, owner: str, motor_name: str,
                  description: str = "",
                  physics: MotorPhysics | None = None) -> MotorState:
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
    global _singleton
    if _singleton is None:
        _singleton = SystemRegistry()
    return _singleton
