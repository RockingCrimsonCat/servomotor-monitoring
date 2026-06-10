"""
Модуль 4: Streamlit GUI системи моніторингу сервомотора (з ролями).

Запуск (з PyCharm Terminal або зовнішнього термінала):
    streamlit run module4_gui.py

Дві ролі користувачів:
- Оператор — моніторить власні мотори (один або декілька).
- Адміністратор — бачить навантаження від усіх операторів та
  перенавчає моделі.

GUI оновлюється автоматично без перезапуску. Працює локально.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from module1_data_generator import stream_step, append_live_row
from module2_ml_models import (
    detect_anomalies,
    load_models,
    models_exist,
    train_models,
)
from module3_diagnostics import build_report
from module5_user_registry import MotorState, get_registry


st.set_page_config(
    page_title="ServoMotor Health Monitoring System",
    page_icon="*",
    layout="wide",
)

REFRESH_INTERVAL_MS = 1500
RECOMMENDATIONS_HOLD_SEC = 5


def _init_session():
    if "user" not in st.session_state:
        st.session_state.user = None
    if "role" not in st.session_state:
        st.session_state.role = None
    if "selected_motor" not in st.session_state:
        st.session_state.selected_motor = None
    if "rec_cache" not in st.session_state:
        st.session_state.rec_cache = {}
    if "rec_state" not in st.session_state:
        st.session_state.rec_state = {}
    if "rec_updated_at" not in st.session_state:
        st.session_state.rec_updated_at = {}


_init_session()


@st.cache_resource(show_spinner="Завантаження ML-моделей...")
def _get_models():
    if not models_exist():
        train_models()
    return load_models()


clf, iforest = _get_models()
registry = get_registry()


def render_login():
    st.markdown(
        "<h1 style='text-align:center;'>ServoMotor Health Monitoring System</h1>",
        unsafe_allow_html=True,
    )
    st.caption(
        "<div style='text-align:center;'>Локальна автономна система моніторингу</div>",
        unsafe_allow_html=True,
    )
    st.write("")

    _, mid, _ = st.columns([1, 1.2, 1])
    with mid:
        st.markdown("### Вхід у систему")
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input(
                "Ім'я користувача",
                placeholder="admin / Ваше ім'я",
            )
            submit = st.form_submit_button("Увійти", use_container_width=True)

        st.caption(
            "🔐 **Адміністратор** входить під ім'ям **`admin`** (єдиний).\n\n"
            "👨‍💻 **Оператори** входять за власним ім'ям. Якщо вашого "
            "імені немає у системі — попросіть адміністратора додати вас."
        )

        if submit:
            name = username.strip()
            if not name:
                st.error("Введіть ім'я користувача.")
                st.stop()
            ok, role_code = registry.can_login(name)
            if not ok:
                st.error(
                    f"❌ Користувача **'{name}'** не зареєстровано в системі. "
                    f"Зверніться до адміністратора, щоб додати ваше ім'я."
                )
                st.stop()
            registry.register_user(name, role_code)
            if role_code == "operator":
                user = registry.users[name]
                if not user.motors:
                    registry.add_motor(name, "Motor-1")
            st.session_state.user = name
            st.session_state.role = role_code
            st.rerun()


if st.session_state.user is None:
    render_login()
    st.stop()


user = registry.users.get(st.session_state.user)
if user is None or (
    st.session_state.role == "operator"
    and st.session_state.user not in registry.registered_operators
):
    st.session_state.user = None
    st.session_state.role = None
    st.session_state.selected_motor = None
    st.error(
        "⚠️ Ваш обліковий запис був видалений адміністратором. "
        "Виконано автоматичний вихід."
    )
    time.sleep(2)
    st.rerun()

role_label = "Адміністратор" if user.role == "admin" else "Оператор"
role_icon = "👑" if user.role == "admin" else "👨‍💻"

top_left, top_right = st.columns([0.78, 0.22])
with top_left:
    st.markdown(
        "<h1 style='margin-bottom:0;'>ServoMotor Health Monitoring System</h1>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"{role_icon} **{user.name}** · {role_label} · автономний режим"
    )
with top_right:
    if st.button("⤴ Вийти", use_container_width=True):
        st.session_state.user = None
        st.session_state.role = None
        st.session_state.selected_motor = None
        st.rerun()

st.divider()


STATE_COLOR_MAP = {
    "NORMAL":  "#2ecc71",
    "WEAR":    "#f1c40f",
    "FAILURE": "#e74c3c",
}
RISK_COLOR_MAP = {"Low": "#2ecc71", "Medium": "#f1c40f", "High": "#e74c3c"}


def _plot(series_name: str, color: str, units: str,
          df: pd.DataFrame, anomaly_mask: np.ndarray) -> go.Figure:
    fig = go.Figure()
    if df.empty:
        return fig
    x_vals = np.arange(len(df))
    y_vals = df[series_name].values
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_vals, mode="lines", name=series_name,
        line=dict(color=color, width=2),
        hovertemplate=f"%{{y:.2f}} {units}<extra></extra>",
    ))
    if anomaly_mask.any():
        a_idx = np.where(anomaly_mask)[0]
        fig.add_trace(go.Scatter(
            x=a_idx, y=y_vals[a_idx], mode="markers", name="Anomaly",
            marker=dict(color="#e74c3c", size=9, symbol="circle",
                        line=dict(color="white", width=1)),
            hovertemplate=f"ANOMALY: %{{y:.2f}} {units}<extra></extra>",
        ))
    fig.update_layout(
        height=240, margin=dict(l=10, r=10, t=20, b=10),
        showlegend=False,
        xaxis_title="Time (samples)",
        yaxis_title=f"{series_name.capitalize()} ({units})",
        template="plotly_dark",
    )
    return fig


def _render_motor_card(m: MotorState, is_selected: bool) -> None:
    state_color = STATE_COLOR_MAP.get(m.last_state, "#888")
    conn_color = "#2ecc71" if m.connected else "#7f8c8d"
    conn_text = "ONLINE" if m.connected else "OFFLINE"
    descr = m.description.strip() if m.description else "—"
    if len(descr) > 30:
        descr = descr[:28] + "…"

    if is_selected:
        outline = "2px solid #ffffff"
        shadow = f"0 0 12px {state_color}"
    else:
        outline = f"1px solid {state_color}33"
        shadow = "none"

    st.markdown(
        f"""
        <div style="padding:10px 12px; border-radius:12px; background:#1f2025;
                    border:{outline}; border-left:5px solid {state_color};
                    box-shadow:{shadow}; margin-bottom:4px;">
            <div style="display:flex; justify-content:space-between;
                       align-items:center;">
                <div style="font-weight:700; font-size:1.05em;
                           color:#f5f5f5;">{m.name}</div>
                <div style="display:flex; align-items:center; gap:6px;">
                    <span style="display:inline-block; width:9px; height:9px;
                                border-radius:50%; background:{conn_color};
                                box-shadow:0 0 5px {conn_color};"></span>
                    <span style="font-size:0.75em; color:{conn_color};
                                font-weight:600;">{conn_text}</span>
                </div>
            </div>
            <div style="font-size:0.8em; color:#aaa; margin-top:3px;
                       overflow:hidden; white-space:nowrap;
                       text-overflow:ellipsis;">
                📍 {descr}
            </div>
            <div style="margin-top:8px; display:flex;
                       justify-content:space-between; align-items:center;">
                <span style="background:{state_color}; color:white;
                            padding:2px 9px; border-radius:8px; font-size:0.8em;
                            font-weight:700; letter-spacing:0.3px;">
                    {m.last_state}
                </span>
                <span style="color:{state_color}; font-weight:700;
                            font-size:0.95em;">
                    P_fail {m.last_failure_prob:.0f}%
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_operator_overview(motors: list[MotorState]) -> None:
    if not motors:
        return

    by_state = {"NORMAL": 0, "WEAR": 0, "FAILURE": 0}
    for m in motors:
        by_state[m.last_state] = by_state.get(m.last_state, 0) + 1
    online_cnt = sum(1 for m in motors if m.connected)

    head_l, head_r = st.columns([0.6, 0.4])
    with head_l:
        st.markdown(
            f"### 🗂 Огляд моїх моторів ({len(motors)})"
        )
    with head_r:
        st.markdown(
            f"""
            <div style="text-align:right; padding-top:14px;
                       font-size:0.9em; color:#bbb;">
                <span style="color:#2ecc71; font-weight:600;">
                    🟢 {by_state['NORMAL']} NORMAL
                </span>
                &nbsp;·&nbsp;
                <span style="color:#f1c40f; font-weight:600;">
                    🟡 {by_state['WEAR']} WEAR
                </span>
                &nbsp;·&nbsp;
                <span style="color:#e74c3c; font-weight:600;">
                    🔴 {by_state['FAILURE']} FAILURE
                </span>
                &nbsp;·&nbsp;
                <span>online: {online_cnt}/{len(motors)}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    PER_ROW = 4
    selected = st.session_state.selected_motor
    for row_start in range(0, len(motors), PER_ROW):
        row = motors[row_start: row_start + PER_ROW]
        cols = st.columns(len(row))
        for col, m in zip(cols, row):
            with col:
                _render_motor_card(m, is_selected=(m.name == selected))
                is_sel = (m.name == selected)
                if st.button(
                    "✓ Обраний" if is_sel else "Обрати",
                    key=f"switch_to_{m.name}",
                    disabled=is_sel,
                    use_container_width=True,
                ):
                    st.session_state.selected_motor = m.name
                    st.rerun()

    st.divider()


def _render_dashboard(motor: MotorState, anomaly_mask: np.ndarray,
                      report, rec_hold_sec: int):
    """Рендерить індикатори + графіки для одного мотора."""
    buffer_df = pd.DataFrame(list(motor.buffer))

    state_col, predict_col, anomaly_col = st.columns([0.32, 0.36, 0.32])

    with state_col:
        st.markdown("### Поточний стан")
        max_conf = max(report.class_probabilities.values(), default=0) * 100
        st.markdown(
            f"""
            <div style="display:flex;align-items:center;justify-content:center;
                        height:160px;border-radius:18px;
                        background:radial-gradient(circle at 50% 40%,
                                    {report.color},#1c1c1c);
                        box-shadow:0 0 22px {report.color};">
                <div style="text-align:center;">
                    <div style="font-size:1.9em;font-weight:700;color:white;
                                text-shadow:0 0 8px rgba(0,0,0,0.7);">
                        {report.state}
                    </div>
                    <div style="color:#f5f5f5;font-size:0.9em;margin-top:6px;">
                        впевненість: {max_conf:.1f}%
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with predict_col:
        st.markdown("### Прогноз відмови")
        risk_color = RISK_COLOR_MAP[report.risk_level]
        st.markdown(
            f"""
            <div style="padding:18px;border-radius:18px;background:#202225;
                        border:1px solid #2e2e2e;">
                <div style="font-size:0.9em;color:#aaa;">Ймовірність відмови</div>
                <div style="font-size:2.0em;font-weight:700;color:{risk_color};">
                    {report.failure_probability:.1f}%
                </div>
                <div style="margin-top:6px;color:#dddddd;">
                    Рівень ризику:
                    <span style="color:{risk_color};font-weight:700;
                                margin-left:4px;">
                        {report.risk_level}
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(min(int(report.failure_probability), 100))

    with anomaly_col:
        st.markdown("### Isolation Forest")
        n_anom = int(anomaly_mask.sum()) if anomaly_mask.size else 0
        st.markdown(
            f"""
            <div style="padding:18px;border-radius:18px;background:#202225;
                        border:1px solid #2e2e2e;">
                <div style="font-size:0.9em;color:#aaa;">Аномалії у вікні</div>
                <div style="font-size:2.0em;font-weight:700;color:#e74c3c;">
                    {n_anom} <span style="font-size:0.55em;color:#aaa;">
                                / {len(buffer_df)}</span>
                </div>
                <div style="margin-top:6px;color:#ccc;">
                    частка: {report.anomaly_rate*100:.1f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    mkey = f"{motor.owner}:{motor.name}"
    now = time.time()
    state_changed = report.state != st.session_state.rec_state.get(mkey)
    elapsed = now - st.session_state.rec_updated_at.get(mkey, 0.0)

    if (state_changed or elapsed >= rec_hold_sec
            or not st.session_state.rec_cache.get(mkey)):
        st.session_state.rec_cache[mkey] = list(report.recommendations)
        st.session_state.rec_state[mkey] = report.state
        st.session_state.rec_updated_at[mkey] = now

    rec_head_l, rec_head_r = st.columns([0.7, 0.3])
    with rec_head_l:
        st.markdown("#### Рекомендації")
    with rec_head_r:
        age = max(0.0, now - st.session_state.rec_updated_at[mkey])
        nxt = max(0.0, rec_hold_sec - age)
        st.caption(
            f"Оновлено {age:.1f} с тому · наступне через {nxt:.1f} с"
        )
    for rec in st.session_state.rec_cache[mkey]:
        st.markdown(f"- {rec}")

    st.divider()

    r1c1, r1c2 = st.columns(2)
    r2c1, r2c2 = st.columns(2)
    with r1c1:
        st.markdown("**Температура (°C)**")
        st.plotly_chart(_plot("temperature", "#ff7f50", "°C", buffer_df, anomaly_mask),
                        use_container_width=True)
    with r1c2:
        st.markdown("**Струм (A)**")
        st.plotly_chart(_plot("current", "#1abc9c", "A", buffer_df, anomaly_mask),
                        use_container_width=True)
    with r2c1:
        st.markdown("**Швидкість (об/хв)**")
        st.plotly_chart(_plot("speed", "#3498db", "rpm", buffer_df, anomaly_mask),
                        use_container_width=True)
    with r2c2:
        st.markdown("**Вібрація (мм/с)**")
        st.plotly_chart(_plot("vibration", "#9b59b6", "mm/s", buffer_df, anomaly_mask),
                        use_container_width=True)


def render_operator_view():
    with st.sidebar:
        st.subheader(f"Мотори оператора {user.name}")
        motors_list = registry.list_motors(user.name)
        if not motors_list:
            registry.add_motor(user.name, "Motor-1")
            motors_list = ["Motor-1"]

        if (st.session_state.selected_motor not in motors_list):
            st.session_state.selected_motor = motors_list[0]

        def _motor_label(name: str) -> str:
            m = registry.get_motor(user.name, name)
            if m and m.description:
                return f"{name} — {m.description}"
            return name

        st.session_state.selected_motor = st.selectbox(
            "Обрати мотор", motors_list,
            index=motors_list.index(st.session_state.selected_motor),
            format_func=_motor_label,
        )

        with st.expander("➕ Додати новий мотор"):
            with st.form("add_motor_form", clear_on_submit=True):
                new_descr = st.text_input(
                    "Опис (локація / позначення)",
                    placeholder="Маніпулятор-1 / Вісь-J1",
                    help="Вільний опис розташування або призначення мотора.",
                )
                add_submit = st.form_submit_button(
                    "Додати мотор", use_container_width=True, type="primary",
                )
            if add_submit:
                next_idx = len(motors_list) + 1
                while f"Motor-{next_idx}" in motors_list:
                    next_idx += 1
                new_name = f"Motor-{next_idx}"
                registry.add_motor(user.name, new_name,
                                   description=new_descr)
                st.session_state.selected_motor = new_name
                st.rerun()

        if st.button("➖ Видалити поточний мотор", use_container_width=True,
                     disabled=len(motors_list) <= 1):
            registry.remove_motor(user.name, st.session_state.selected_motor)
            st.session_state.selected_motor = None
            st.rerun()

        st.divider()
        motor = registry.get_motor(user.name, st.session_state.selected_motor)

        st.subheader(f"Налаштування «{motor.name}»")

        edited_descr = st.text_area(
            "Опис мотора",
            value=motor.description,
            placeholder="Маніпулятор-1 / Вісь-J1",
            height=70,
            help="Натисніть поза полем — опис збережеться автоматично.",
        )
        if edited_descr != motor.description:
            registry.update_motor_description(
                user.name, motor.name, edited_descr,
            )

        with st.expander("⚙ Фізичний профіль мотора"):
            p = motor.physics
            st.markdown(
                f"- **Напруга:** {p.voltage:.1f} В\n"
                f"- **Опір обмотки:** {p.resistance:.3f} Ом\n"
                f"- **Постійна моменту k_t:** {p.torque_constant:.4f}\n"
                f"- **Постійна ЕРС k_e:** {p.back_emf_constant:.4f}\n"
                f"- **Ном. навантаження:** {p.load_torque:.3f} Н·м\n"
                f"- **Темп. оточення:** {p.ambient_temp:.1f} °C\n"
                f"- **Термоопір:** {p.thermal_resistance:.3f} °C/Вт\n"
                f"- **Шум-фактор:** ×{p.noise_scale:.2f}"
            )
            st.caption(
                "Кожен мотор отримує власний фізичний профіль з "
                "відхиленнями ±5–10% від номіналу — це імітує "
                "природний розкид параметрів серійного виробництва."
            )

        motor.mode = st.radio(
            "Імітований режим",
            ["NORMAL", "WEAR", "FAILURE"],
            index=["NORMAL", "WEAR", "FAILURE"].index(motor.mode),
            key=f"mode_{motor.name}",
        )
        motor.connected = st.toggle(
            "Підключення до мотора",
            value=motor.connected,
            key=f"conn_{motor.name}",
        )
        motor.points_per_tick = st.slider(
            "Точок за оновлення", 1, 10, motor.points_per_tick,
            key=f"pts_{motor.name}",
            help="Скільки нових точок телеметрії генерувати за одне "
                 "оновлення (1.5 с). Значення індивідуальне для цього мотора.",
        )
        motor.rec_hold_sec = st.slider(
            "Витримка рекомендацій, с", 2, 15, motor.rec_hold_sec,
            key=f"rec_{motor.name}",
            help="Як довго тримати поточний текст порад перед оновленням. "
                 "Зміна стану оновлює список миттєво. "
                 "Значення індивідуальне для цього мотора.",
        )

        st.caption(
            f"Графіки оновлюються кожні {REFRESH_INTERVAL_MS} мс, "
            f"рекомендації — раз на {motor.rec_hold_sec} с."
        )

    selected_name = st.session_state.selected_motor
    motor = None
    anomaly_mask = np.array([], dtype=bool)
    report = None
    all_motors: list[MotorState] = []

    for motor_name in motors_list:
        m = registry.get_motor(user.name, motor_name)
        if m is None:
            continue
        all_motors.append(m)

        if m.connected:
            for _ in range(m.points_per_tick):
                row = stream_step(m.stream_state, m.mode)
                m.buffer.append(row)
                append_live_row(row, owner=user.name, motor_name=m.name)
            m.tick_count += m.points_per_tick
            m.last_update_at = time.time()

        buffer_df_m = pd.DataFrame(list(m.buffer))
        if not buffer_df_m.empty:
            a_mask_m, _ = detect_anomalies(iforest, buffer_df_m)
            rep_m = build_report(buffer_df_m, clf, a_mask_m)
        else:
            a_mask_m = np.array([], dtype=bool)
            rep_m = build_report(buffer_df_m, clf, a_mask_m)

        prev_state = m.last_state
        if prev_state != rep_m.state:
            if rep_m.state == "FAILURE":
                st.toast(
                    f"🚨 {m.name}: КРИТИЧНИЙ СТАН (FAILURE) — "
                    f"P_fail {rep_m.failure_probability:.0f}%",
                    icon="🚨",
                )
            elif rep_m.state == "WEAR":
                st.toast(
                    f"⚠️ {m.name}: виявлено знос",
                    icon="⚠️",
                )
            elif prev_state in ("WEAR", "FAILURE") and rep_m.state == "NORMAL":
                st.toast(
                    f"✅ {m.name}: стан нормалізувався",
                    icon="✅",
                )

        m.last_anomalies = int(a_mask_m.sum()) if a_mask_m.size else 0
        m.last_state = rep_m.state
        m.last_failure_prob = float(rep_m.failure_probability)

        if m.name == selected_name:
            motor = m
            anomaly_mask = a_mask_m
            report = rep_m

    if motor is None and all_motors:
        motor = all_motors[0]
        st.session_state.selected_motor = motor.name
        buffer_df_m = pd.DataFrame(list(motor.buffer))
        if not buffer_df_m.empty:
            anomaly_mask, _ = detect_anomalies(iforest, buffer_df_m)
            report = build_report(buffer_df_m, clf, anomaly_mask)
        else:
            report = build_report(buffer_df_m, clf, anomaly_mask)

    _render_operator_overview(all_motors)

    h_left, h_right = st.columns([0.75, 0.25])
    with h_left:
        st.markdown(f"## Мотор: `{motor.name}`")
        if motor.description:
            st.caption(f"📍 {motor.description}")
    with h_right:
        cc = "#2ecc71" if motor.connected else "#7f8c8d"
        ct = "ONLINE" if motor.connected else "OFFLINE"
        st.markdown(
            f"""
            <div style="text-align:right;padding-top:18px;">
                <span style="display:inline-block;width:14px;height:14px;
                            border-radius:50%;background:{cc};
                            box-shadow:0 0 8px {cc};margin-right:8px;"></span>
                <span style="font-weight:600;color:{cc};">{ct}</span>
                <div style="font-size:0.85em;color:#888;margin-top:4px;">
                    тіків: {motor.tick_count}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()
    _render_dashboard(motor, anomaly_mask, report, motor.rec_hold_sec)


def render_admin_view():
    with st.sidebar:
        st.subheader("Панель адміністратора")
        st.markdown(f"**Користувач:** {user.name} 👑")
        st.markdown(
            "**Привілеї:**\n"
            "- ✅ Додавання/видалення операторів\n"
            "- ✅ Перегляд усіх моторів та навантаження\n"
            "- ✅ Перегляд аудит-логу системи"
        )
        st.divider()

        st.subheader("👥 Управління операторами")

        with st.form("add_op_form", clear_on_submit=True):
            new_op_name = st.text_input(
                "Ім'я нового оператора",
                placeholder="Іван, Олена, Петро...",
            )
            add_btn = st.form_submit_button(
                "➕ Додати оператора", use_container_width=True,
            )
        if add_btn:
            n = new_op_name.strip()
            if not n:
                st.warning("Введіть ім'я.")
            elif n == registry.ADMIN_NAME:
                st.error(
                    f"Ім'я '{registry.ADMIN_NAME}' зарезервоване для адміна."
                )
            elif n in registry.registered_operators:
                st.warning(f"Оператор '{n}' уже існує.")
            else:
                try:
                    registry.create_operator(n)
                    st.success(f"Оператора **{n}** додано.")
                    time.sleep(0.6)
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))

        ops = registry.list_registered_operators()
        if not ops:
            st.caption("— операторів ще не зареєстровано —")
        else:
            st.markdown("**Зареєстровані оператори:**")
            for op_name in ops:
                online = op_name in registry.users
                status_emoji = "🟢" if online else "⚪"
                c1, c2 = st.columns([0.75, 0.25])
                with c1:
                    st.markdown(f"{status_emoji} **{op_name}**")
                with c2:
                    if st.button(
                        "➖", key=f"rm_op_{op_name}",
                        help=f"Видалити оператора '{op_name}'",
                        use_container_width=True,
                    ):
                        registry.remove_operator(op_name)
                        st.rerun()
            st.caption("🟢 — онлайн, ⚪ — офлайн")

        st.divider()
        st.caption(
            f"Графіки оновлюються кожні {REFRESH_INTERVAL_MS} мс."
        )

    totals = registry.system_totals()
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Операторів онлайн", totals["operators"])
    m2.metric("Усього моторів", totals["total_motors"])
    m3.metric("Активних моторів", totals["active_motors"])
    m4.metric("Сумарні тіки", totals["total_ticks"])

    registry.sample_system_metrics()
    sys_df = registry.get_system_metrics_df()
    latest = registry.latest_system_metric()

    st.markdown("### 💻 Навантаження від процесу системи")
    if latest is None or sys_df.empty:
        st.info(
            "Збір системних метрик розпочато — графіки з'являться "
            "через декілька секунд."
        )
    else:
        cur_cpu, cur_mem_pct = latest["cpu_pct"], latest["mem_pct"]
        cur_mem_used = latest["mem_used_mb"]
        cur_mem_total = latest["mem_total_mb"]

        cpu_color = (
            "#2ecc71" if cur_cpu < 30
            else "#f1c40f" if cur_cpu < 60
            else "#e74c3c"
        )
        mem_color = (
            "#2ecc71" if cur_mem_pct < 10
            else "#f1c40f" if cur_mem_pct < 25
            else "#e74c3c"
        )
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(
                f"""
                <div style="padding:14px;border-radius:14px;background:#202225;
                            border:1px solid #2e2e2e;">
                  <div style="font-size:0.9em;color:#aaa;">CPU процесу</div>
                  <div style="font-size:2.0em;font-weight:700;color:{cpu_color};">
                      {cur_cpu:.1f}%
                  </div>
                  <div style="font-size:0.8em;color:#888;margin-top:4px;">
                      від усіх логічних ядер
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with c2:
            st.markdown(
                f"""
                <div style="padding:14px;border-radius:14px;background:#202225;
                            border:1px solid #2e2e2e;">
                  <div style="font-size:0.9em;color:#aaa;">RAM процесу</div>
                  <div style="font-size:2.0em;font-weight:700;color:{mem_color};">
                      {cur_mem_used:,.0f} МБ
                  </div>
                  <div style="font-size:0.8em;color:#888;margin-top:4px;">
                      {cur_mem_pct:.2f}% від системних {cur_mem_total:,.0f} МБ
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        x_idx = np.arange(len(sys_df))
        gc1, gc2 = st.columns(2)
        with gc1:
            fig_cpu = go.Figure()
            fig_cpu.add_trace(go.Scatter(
                x=x_idx, y=sys_df["cpu_pct"],
                mode="lines", line=dict(color="#3498db", width=2),
                fill="tozeroy", fillcolor="rgba(52,152,219,0.15)",
                hovertemplate="%{y:.1f}%<extra></extra>",
            ))
            fig_cpu.update_layout(
                height=240, template="plotly_dark",
                margin=dict(l=10, r=10, t=30, b=10),
                title="CPU процесу, % від усіх ядер",
                yaxis=dict(title="%", rangemode="tozero"),
                xaxis_title="Sample",
                showlegend=False,
            )
            st.plotly_chart(fig_cpu, use_container_width=True)

        with gc2:
            fig_mem = go.Figure()
            fig_mem.add_trace(go.Scatter(
                x=x_idx, y=sys_df["mem_used_mb"],
                mode="lines", line=dict(color="#9b59b6", width=2),
                fill="tozeroy", fillcolor="rgba(155,89,182,0.15)",
                hovertemplate="%{y:,.0f} МБ<extra></extra>",
            ))
            fig_mem.update_layout(
                height=240, template="plotly_dark",
                margin=dict(l=10, r=10, t=30, b=10),
                title="RAM процесу, МБ (RSS)",
                yaxis=dict(title="МБ", rangemode="tozero"),
                xaxis_title="Sample",
                showlegend=False,
            )
            st.plotly_chart(fig_mem, use_container_width=True)

    st.divider()

    st.markdown("### 🚨 Критичні мотори (FAILURE)")
    failing = registry.get_failing_motors()
    if not failing:
        st.success("Жоден мотор не перебуває у стані FAILURE.")
    else:
        for op, mot, p in failing:
            st.error(
                f"**{op} / {mot}** — ймовірність відмови **{p:.1f}%**",
                icon="🔥",
            )

    st.divider()

    st.markdown("### 📊 Навантаження на систему по операторах")
    df = registry.get_load_summary()
    if df.empty:
        st.info(
            "Жоден оператор не зареєстрований. "
            "Додайте операторів у бічній панелі ліворуч, після чого "
            "вони зможуть увійти за своїми іменами."
        )
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)

        if "Тіків/с" in df.columns and len(df):
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df["Оператор"], y=df["Тіків/с"],
                marker_color="#3498db",
                hovertemplate="%{x}: %{y:.2f} тіків/с<extra></extra>",
            ))
            fig.update_layout(
                height=300, template="plotly_dark",
                margin=dict(l=20, r=20, t=30, b=20),
                title="Інтенсивність генерації даних, тіків/с",
                xaxis_title="Оператор",
                yaxis_title="Тіки/с",
            )
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.markdown("### 🔧 Усі мотори у системі")
    rows = []
    for opname in registry.list_operators():
        opdata = registry.users[opname]
        for m in opdata.motors.values():
            rows.append({
                "Оператор": opname,
                "Мотор": m.name,
                "Опис": m.description or "—",
                "Підключення": "ONLINE" if m.connected else "OFFLINE",
                "Поточний стан": m.last_state,
                "P_fail (%)": round(m.last_failure_prob, 1),
                "Аномалій у вікні": m.last_anomalies,
                "Тіків": m.tick_count,
                "Режим джерела": m.mode,
            })
    if not rows:
        st.caption("— немає зареєстрованих моторів —")
    else:
        st.dataframe(pd.DataFrame(rows),
                     use_container_width=True, hide_index=True)

        with st.expander("📥 Завантажити CSV-лог окремого мотора"):
            options = [
                (r["Оператор"], r["Мотор"]) for r in rows
            ]
            if options:
                sel = st.selectbox(
                    "Оберіть мотор",
                    options=range(len(options)),
                    format_func=lambda i: f"{options[i][0]} / {options[i][1]}",
                )
                op, mot = options[sel]
                from module1_data_generator import stream_path
                import os as _os
                path = stream_path(op, mot)
                if _os.path.exists(path):
                    size = _os.path.getsize(path)
                    with open(path, "rb") as f:
                        st.download_button(
                            label=f"⬇ Завантажити {_os.path.basename(path)} "
                                  f"({size//1024} KB)",
                            data=f.read(),
                            file_name=_os.path.basename(path),
                            mime="text/csv",
                            use_container_width=True,
                        )
                else:
                    st.caption("Файл потоку ще не створено "
                               "(мотор не згенерував жодної точки).")

    st.divider()

    st.markdown("### 📜 Аудит-лог (останні 30 подій)")
    audit_df = registry.get_audit_log(limit=30)
    if audit_df.empty:
        st.caption("— подій ще не зафіксовано —")
    else:
        st.dataframe(audit_df, use_container_width=True, hide_index=True)


if user.role == "admin":
    render_admin_view()
else:
    render_operator_view()


_autorefresh_done = False
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=REFRESH_INTERVAL_MS, key="auto_refresh")
    _autorefresh_done = True
except Exception:
    pass

if not _autorefresh_done:
    time.sleep(REFRESH_INTERVAL_MS / 1000.0)
    st.rerun()
