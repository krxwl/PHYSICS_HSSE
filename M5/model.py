"""Физические свойства и уравнения движения маятника."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PhysicalPendulum:
    """Тонкий однородный стержень и сплошной шар на его конце."""

    gravity: float
    rod_mass: float
    sphere_mass: float
    length: float
    sphere_radius: float

    def __post_init__(self):
        values = {
            "gravity": self.gravity,
            "rod_mass": self.rod_mass,
            "sphere_mass": self.sphere_mass,
            "length": self.length,
            "sphere_radius": self.sphere_radius,
        }
        for name, value in values.items():
            if value <= 0:
                raise ValueError(f"{name} должно быть положительным")

    @property
    def inertia(self) -> float:
        """Момент инерции относительно точки подвеса, кг·м²."""
        rod_inertia = self.rod_mass * self.length**2 / 3
        sphere_inertia = (
            self.sphere_mass * self.length**2
            + 2 * self.sphere_mass * self.sphere_radius**2 / 5
        )
        return rod_inertia + sphere_inertia

    @property
    def gravity_moment_coefficient(self) -> float:
        """Сумма m_i r_i для гравитационного момента, кг·м."""
        return self.rod_mass * self.length / 2 + self.sphere_mass * self.length

    @property
    def small_angle_frequency(self) -> float:
        """Угловая частота малых колебаний, рад/с."""
        return np.sqrt(self.gravity * self.gravity_moment_coefficient / self.inertia)

    @property
    def small_angle_period(self) -> float:
        """Период малых колебаний, с."""
        return 2 * np.pi / self.small_angle_frequency

    def angular_acceleration(
        self,
        angle: float,
        angular_velocity: float,
        damping: float = 0.0,
    ) -> float:
        """Угловое ускорение с вязким моментом -damping * omega."""
        torque = (
            -damping * angular_velocity
            - self.gravity * self.gravity_moment_coefficient * np.sin(angle)
        )
        return torque / self.inertia

    def energy(self, angle, angular_velocity):
        """Механическая энергия"""
        kinetic = 0.5 * self.inertia * np.asarray(angular_velocity) ** 2
        potential = self.gravity * self.gravity_moment_coefficient * (1 - np.cos(angle))
        return kinetic + potential
