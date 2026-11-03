"""Сценарий исследования физического маятника.

Запуск из этой папки: python physical_pendulum.py
"""

from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from constants import L, M_ROD, M_SPHERE, R, g, DAMPING
from energy import analyze_energy
from model import PhysicalPendulum
from plotting import plot_results
from simulation import Motion, numerical_period, simulate
from theory import (
    damping_frequencies,
    exact_period,
    exact_undamped_angle,
    linear_damped_angle,
    small_angle_undamped_angle,
)


@dataclass(frozen=True)
class FreeMotionStudy:
    amplitude: float
    motion: Motion
    exact_angle: np.ndarray
    small_angle: np.ndarray
    exact_period: float
    max_energy_drift: float
    max_angle_error: float


@dataclass(frozen=True)
class PeriodStudy:
    amplitudes_degrees: np.ndarray
    numerical_periods: np.ndarray
    theoretical_periods: np.ndarray


@dataclass(frozen=True)
class DampedMotionStudy:
    damping: float
    amplitude: float
    motion: Motion
    theoretical_angle: np.ndarray
    theoretical_period: float
    numerical_period: float
    energy_balance_error: float


def create_pendulum() -> PhysicalPendulum:
    """Создаёт модель из параметров constants.py."""
    return PhysicalPendulum(
        gravity=g,
        rod_mass=M_ROD,
        sphere_mass=M_SPHERE,
        length=L,
        sphere_radius=R,
    )


def study_free_motion(pendulum, amplitude_degrees=60, periods_to_simulate=8):
    """Считает свободные колебания и проверяет их по теории и энергии."""
    amplitude = np.deg2rad(amplitude_degrees)
    theoretical_period = exact_period(pendulum, amplitude)
    motion = simulate(
        pendulum,
        initial_angle=amplitude,
        duration=periods_to_simulate * theoretical_period,
        damping=0.0,
    )

    exact_angle = exact_undamped_angle(pendulum, motion.time, amplitude)
    small_angle = small_angle_undamped_angle(pendulum, motion.time, amplitude)
    energy = analyze_energy(pendulum, motion)

    return FreeMotionStudy(
        amplitude=amplitude,
        motion=motion,
        exact_angle=exact_angle,
        small_angle=small_angle,
        exact_period=theoretical_period,
        max_energy_drift=energy.max_relative_drift,
        max_angle_error=float(np.max(np.abs(motion.angle - exact_angle))),
    )


def study_period_vs_amplitude(pendulum, amplitudes_degrees=None):
    """Сравнивает численный и теоретический периоды для разных амплитуд."""
    if amplitudes_degrees is None:
        amplitudes_degrees = np.array([5, 10, 20, 40, 60, 90, 120, 150])
    else:
        amplitudes_degrees = np.asarray(amplitudes_degrees, dtype=float)

    amplitudes_radians = np.deg2rad(amplitudes_degrees)
    theoretical = np.array(
        [exact_period(pendulum, angle) for angle in amplitudes_radians]
    )
    numerical = np.array(
        [
            numerical_period(pendulum, angle, period)
            for angle, period in zip(amplitudes_radians, theoretical)
        ]
    )

    return PeriodStudy(
        amplitudes_degrees=amplitudes_degrees,
        numerical_periods=numerical,
        theoretical_periods=theoretical,
    )


def study_damped_motion(
    pendulum,
    damping,
    amplitude_degrees=20,
    periods_to_simulate=12,
):
    """Считает затухающие колебания и сравнивает с линейной теорией."""
    amplitude = np.deg2rad(amplitude_degrees)
    _, omega_d = damping_frequencies(pendulum, damping)
    theoretical_period = 2 * np.pi / omega_d
    motion = simulate(
        pendulum,
        initial_angle=amplitude,
        duration=periods_to_simulate * theoretical_period,
        damping=damping,
        samples=6000,
    )

    theoretical_angle = linear_damped_angle(pendulum, motion.time, amplitude, damping)
    energy = analyze_energy(pendulum, motion, damping)

    crossings = motion.upward_crossings
    numerical_period = (
        float(np.mean(np.diff(crossings))) if crossings.size >= 2 else float("nan")
    )

    return DampedMotionStudy(
        damping=damping,
        amplitude=amplitude,
        motion=motion,
        theoretical_angle=theoretical_angle,
        theoretical_period=theoretical_period,
        numerical_period=numerical_period,
        energy_balance_error=energy.max_balance_error,
    )


def print_report(pendulum, free_study, period_study, damped_study):
    """Печатает параметры модели и результаты сравнений."""
    print(f"I = {pendulum.inertia:.6f} кг·м²")
    print(f"B = {pendulum.gravity_moment_coefficient:.6f} кг·м")
    print(f"Период малых колебаний T0 = {pendulum.small_angle_period:.6f} с")
    print("\nСвободные колебания, амплитуда 60°:")
    print(f"Точный нелинейный период = {free_study.exact_period:.6f} с")
    print(
        "Максимальная ошибка theta относительно точного решения = "
        f"{free_study.max_angle_error:.3e} рад"
    )
    print(
        "Максимальное относительное изменение энергии = "
        f"{free_study.max_energy_drift:.3e}"
    )

    print("\nПериод в зависимости от амплитуды:")
    print("Амплитуда     T численный, с     T теория, с      Ошибка, %")
    for angle, numerical, theoretical in zip(
        period_study.amplitudes_degrees,
        period_study.numerical_periods,
        period_study.theoretical_periods,
    ):
        error_percent = 100 * (numerical - theoretical) / theoretical
        print(
            f"{angle:8.1f}°     {numerical:13.6f}     "
            f"{theoretical:12.6f}     {error_percent:9.4f}"
        )

    print("\nЗатухающие колебания, амплитуда 20°:")
    print(f"Период численный = {damped_study.numerical_period:.6f} с")
    print(f"Период линейной теории = {damped_study.theoretical_period:.6f} с")
    print(
        "Ошибка баланса энергии с учётом потерь = "
        f"{damped_study.energy_balance_error:.3e}"
    )


def main():
    pendulum = create_pendulum()
    free_study = study_free_motion(pendulum)
    period_study = study_period_vs_amplitude(pendulum)
    damped_study = study_damped_motion(pendulum, DAMPING)

    print_report(pendulum, free_study, period_study, damped_study)

    figure = plot_results(
        pendulum=pendulum,
        free_motion=free_study.motion,
        free_exact_angle=free_study.exact_angle,
        free_small_angle=free_study.small_angle,
        amplitudes_degrees=period_study.amplitudes_degrees,
        periods_numerical=period_study.numerical_periods,
        damped_motion=damped_study.motion,
        damped_theory_angle=damped_study.theoretical_angle,
        damping=damped_study.damping,
    )
    plt.show()
    return figure


if __name__ == "__main__":
    main()
