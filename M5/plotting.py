"""Построение графиков результатов моделирования."""

import numpy as np
import matplotlib.pyplot as plt

from energy import analyze_energy
from model import PhysicalPendulum
from simulation import Motion
from theory import exact_period


def plot_results(
    pendulum: PhysicalPendulum,
    free_motion: Motion,
    free_exact_angle,
    free_small_angle,
    amplitudes_degrees,
    periods_numerical,
    damped_motion: Motion,
    damped_theory_angle,
    damping: float,
):
    """Строит общий рисунок с четырьмя графиками и возвращает его."""
    free_energy = analyze_energy(pendulum, free_motion)
    damped_energy = analyze_energy(pendulum, damped_motion, damping)

    figure, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.plot(free_motion.time, np.rad2deg(free_motion.angle), label="Численное решение")
    ax.plot(
        free_motion.time,
        np.rad2deg(free_exact_angle),
        "--",
        label="Точное нелинейное решение",
    )
    ax.plot(
        free_motion.time,
        np.rad2deg(free_small_angle),
        ":",
        label="Приближение малых углов",
    )
    ax.set(
        xlabel="Время, с",
        ylabel="Угол, градусы",
        title="Свободные колебания без трения",
    )
    ax.grid(True)
    ax.legend()

    ax = axes[0, 1]
    angle_grid_degrees = np.linspace(1, 150, 300)
    angle_grid_radians = np.deg2rad(angle_grid_degrees)
    period_grid = np.array(
        [exact_period(pendulum, angle) for angle in angle_grid_radians]
    )
    ax.plot(angle_grid_degrees, period_grid, label="Точная нелинейная теория")
    ax.scatter(
        amplitudes_degrees, periods_numerical, marker="o", label="Численный период"
    )
    ax.axhline(
        pendulum.small_angle_period,
        color="gray",
        linestyle=":",
        label="Период малых колебаний",
    )
    ax.set(
        xlabel="Амплитуда, градусы",
        ylabel="Период, с",
        title="Зависимость периода от амплитуды",
    )
    ax.grid(True)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(
        free_motion.time,
        free_energy.energy / free_energy.energy[0],
        label="Без трения: E(t) / E(0)",
    )
    ax.plot(
        damped_motion.time,
        damped_energy.energy / damped_energy.energy[0],
        label="С трением: E(t) / E(0)",
    )
    ax.plot(
        damped_motion.time,
        damped_energy.normalized_balance,
        "--",
        label="С трением: (E + потери) / E(0)",
    )
    ax.set(xlabel="Время, с", ylabel="Относительная энергия", title="Проверка энергии")
    ax.grid(True)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(
        damped_motion.time,
        np.rad2deg(damped_motion.angle),
        label="Численное нелинейное решение",
    )
    ax.plot(
        damped_motion.time,
        np.rad2deg(damped_theory_angle),
        "--",
        label="Линейная теория",
    )
    envelope = np.rad2deg(
        abs(damped_motion.angle[0])
        * np.exp(-damping * damped_motion.time / (2 * pendulum.inertia))
    )
    ax.plot(damped_motion.time, envelope, ":", color="gray", label="Огибающая")
    ax.plot(damped_motion.time, -envelope, ":", color="gray")
    ax.set(
        xlabel="Время, с",
        ylabel="Угол, градусы",
        title="Затухающие свободные колебания",
    )
    ax.grid(True)
    ax.legend()

    figure.tight_layout()
    return figure
