"""Теоретические решения и формулы для сравнения с численным расчётом."""

import numpy as np
from scipy.special import ellipj, ellipk

from model import PhysicalPendulum


def exact_period(pendulum: PhysicalPendulum, amplitude: float) -> float:
    """Нелинейный период без трения через эллиптический интеграл."""
    amplitude = abs(amplitude)
    if amplitude >= np.pi:
        raise ValueError("Для колебаний амплитуда должна быть меньше pi рад")

    modulus = np.sin(amplitude / 2)
    parameter = modulus**2
    return (
        4
        * np.sqrt(
            pendulum.inertia / (pendulum.gravity * pendulum.gravity_moment_coefficient)
        )
        * ellipk(parameter)
    )


def exact_undamped_angle(pendulum, time, amplitude):
    """Точное нелинейное theta(t) при старте из крайнего положения."""
    modulus = np.sin(abs(amplitude) / 2)
    parameter = modulus**2
    quarter_period_factor = ellipk(parameter)
    sn, _, _, _ = ellipj(
        quarter_period_factor + pendulum.small_angle_frequency * time,
        parameter,
    )
    sign = 1.0 if amplitude >= 0 else -1.0
    return sign * 2 * np.arcsin(modulus * sn)


def small_angle_undamped_angle(pendulum, time, amplitude):
    """Линейное решение без трения при sin(theta) ~= theta."""
    return amplitude * np.cos(pendulum.small_angle_frequency * time)


def damping_frequencies(pendulum, damping):
    """Возвращает beta и omega_d для линейного недозатухающего режима."""
    if damping < 0:
        raise ValueError("Коэффициент трения не может быть отрицательным")

    beta = damping / (2 * pendulum.inertia)
    omega_d_squared = pendulum.small_angle_frequency**2 - beta**2
    if omega_d_squared <= 0:
        raise ValueError("Линейная модель не является недозатухающей")

    return beta, np.sqrt(omega_d_squared)


def linear_damped_angle(pendulum, time, amplitude, damping):
    """Линейное решение с вязким трением и нулевой начальной скоростью."""
    beta, omega_d = damping_frequencies(pendulum, damping)
    return (
        amplitude
        * np.exp(-beta * time)
        * (np.cos(omega_d * time) + beta / omega_d * np.sin(omega_d * time))
    )
