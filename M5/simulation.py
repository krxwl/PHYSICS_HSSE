"""Численное интегрирование уравнения движения и оценка периода."""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import solve_ivp

from model import PhysicalPendulum


@dataclass(frozen=True)
class Motion:
    """Временной ряд решения уравнения движения."""

    time: np.ndarray
    angle: np.ndarray
    angular_velocity: np.ndarray
    upward_crossings: np.ndarray


def _equilibrium_crossing(time, state):
    """Событие прохождения положения равновесия theta = 0."""
    return state[0]


_equilibrium_crossing.direction = 1
_equilibrium_crossing.terminal = False


def simulate(
    pendulum: PhysicalPendulum,
    initial_angle: float,
    duration: float,
    damping: float = 0.0,
    samples: int = 5000,
) -> Motion:
    """Решает ОДУ для заданных начальных условий и коэффициента трения."""
    if duration <= 0:
        raise ValueError("Длительность расчёта должна быть положительной")
    if samples < 2:
        raise ValueError("Для расчёта нужны как минимум две точки времени")
    if damping < 0:
        raise ValueError("Коэффициент трения не может быть отрицательным")

    time_grid = np.linspace(0.0, duration, samples)

    def right_hand_side(time, state):
        angle, angular_velocity = state
        return [
            angular_velocity,
            pendulum.angular_acceleration(angle, angular_velocity, damping),
        ]

    solution = solve_ivp(
        fun=right_hand_side,
        t_span=(0.0, duration),
        y0=[initial_angle, 0.0],
        method="DOP853",
        t_eval=time_grid,
        events=_equilibrium_crossing,
        rtol=1e-10,
        atol=1e-12,
        max_step=pendulum.small_angle_period / 100,
    )

    if not solution.success:
        raise RuntimeError(solution.message)

    return Motion(
        time=solution.t,
        angle=solution.y[0],
        angular_velocity=solution.y[1],
        upward_crossings=solution.t_events[0],
    )


def numerical_period(
    pendulum: PhysicalPendulum,
    amplitude: float,
    expected_period: float,
) -> float:
    """Оценивает период по двум однонаправленным проходам через равновесие."""
    motion = simulate(
        pendulum=pendulum,
        initial_angle=amplitude,
        duration=2.05 * expected_period,
        damping=0.0,
        samples=300,
    )
    if motion.upward_crossings.size < 2:
        raise RuntimeError("Для оценки периода найдено меньше двух пересечений")
    return motion.upward_crossings[1] - motion.upward_crossings[0]
