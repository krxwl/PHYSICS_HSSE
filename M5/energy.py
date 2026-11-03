"""Расчёт энергии и проверка энергетического баланса."""

from dataclasses import dataclass

import numpy as np
from scipy.integrate import cumulative_trapezoid

from model import PhysicalPendulum
from simulation import Motion


@dataclass(frozen=True)
class EnergyAnalysis:
    energy: np.ndarray
    dissipated_energy: np.ndarray
    normalized_balance: np.ndarray
    max_relative_drift: float
    max_balance_error: float


def analyze_energy(
    pendulum: PhysicalPendulum,
    motion: Motion,
    damping: float = 0.0,
) -> EnergyAnalysis:
    """Считает E(t), потери b*omega^2 и нормированный баланс энергии."""
    energy = pendulum.energy(motion.angle, motion.angular_velocity)
    dissipated = cumulative_trapezoid(
        damping * motion.angular_velocity**2,
        motion.time,
        initial=0.0,
    )
    normalized_energy = energy / energy[0]
    normalized_balance = (energy + dissipated) / energy[0]

    return EnergyAnalysis(
        energy=energy,
        dissipated_energy=dissipated,
        normalized_balance=normalized_balance,
        max_relative_drift=float(np.max(np.abs(normalized_energy - 1))),
        max_balance_error=float(np.max(np.abs(normalized_balance - 1))),
    )
