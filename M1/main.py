from numpy import ndarray, array  # быстрый массивчик

from scipy.integrate import solve_ivp  # обертка решения диффура
from scipy.constants import g

from math import cos, sin, radians, sqrt

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator  # для изменения масштаба

from constants import *


# функция, показывающая, когда камень упадет
def hit_ground(t: float, coordinates: ndarray[float]) -> float:
    """
    функция которая когда тело упадет на землю (y=0) перестает
    интегрировать диффур на остальном интервале
    """
    return coordinates[1]


hit_ground.terminal = True
hit_ground.direction = 1


class Solver:
    def __init__(self):
        (
            self.alpha,
            self.v_0,
            self.resistance_coefficient,
            self.formula_choice_tag,
            self.weight,
        ) = self.read_data()

        if self.formula_choice_tag is False:
            self.visualise_data(self.get_viscous_resistance_data())
        else:
            self.visualise_data(self.get_frontal_resistance_data())

    def read_data(self) -> tuple[float, float, float, bool, float]:
        """
        считывает начальные условия
        """
        alpha: float = float(input(INPUT_ALPHA_STR))
        v_0: float = float(input(INPUT_INITIAL_SPEED_STR))
        resistance_coefficient: float = float(input(INPUT_COEFFICIENT_STR))

        formula_choice_flag: bool = bool(int(input(RESISTANCY_FORMULA_CHOICE_STR)))
        weight: float = float(input(INPUT_WEIGHT_STR))

        return tuple([alpha, v_0, resistance_coefficient, formula_choice_flag, weight])

    # coords - текущая точка в которой мы находимся
    def speed_equation(self, t: float, coordinates: ndarray[float]) -> list[float]:
        """
        решаем диффур для вязкого трения
        """
        x, y, v_x, v_y = coordinates

        # вычисляем производные в точке
        dvx_dt = -self.resistance_coefficient / self.weight * v_x
        dvy_dt = -g - self.resistance_coefficient / self.weight * v_y

        dx_dt = v_x
        dy_dt = v_y

        return array([dx_dt, dy_dt, dvx_dt, dvy_dt])

    def quad_speed_equation(self, t: float, coordinates: ndarray[float]) -> list[float]:
        """
        решаем диффур для вязкого трения
        """
        x, y, v_x, v_y = coordinates

        # вычисляем производные в точке
        dvx_dt = (
            -self.resistance_coefficient / self.weight * v_x * sqrt(v_x**2 + v_y**2)
        )
        dvy_dt = -g - self.resistance_coefficient / self.weight * v_y * sqrt(
            v_x**2 + v_y**2
        )

        dx_dt = v_x
        dy_dt = v_y

        return array([dx_dt, dy_dt, dvx_dt, dvy_dt])

    def get_start_conditions(self) -> list[float]:
        """
        функция возвращает начальное положение тела
        и проекции начальной скорости
        """
        x0 = 0.0
        y0 = 0.00000001  # чтобы интегрирование не закончилось сразу же
        v_x0 = self.v_0 * cos(radians(self.alpha))
        v_y0 = self.v_0 * sin(radians(self.alpha))
        return array([x0, y0, v_x0, v_y0])

    def get_viscous_resistance_data(self) -> ndarray[float]:
        # решаем задачу коши
        result = solve_ivp(
            fun=self.speed_equation,
            t_span=(0, SPAN_MAX_UPPER_BORDER),  # область интегрирования
            y0=self.get_start_conditions(),  # начальные условия (x0, y0, v_x0, v_y0)
            events=hit_ground,  #  перестанет вычислять диффур на остальной области определения после того как событие произойдет
            dense_output=True,  # решение непрерывно
        )
        return result.y

    def get_frontal_resistance_data(self) -> ndarray[float]:
        # решаем задачу коши
        result = solve_ivp(
            fun=self.quad_speed_equation,
            t_span=(0, SPAN_MAX_UPPER_BORDER),  # область интегрирования
            y0=self.get_start_conditions(),  # начальные условия (x0, y0, v_x0, v_y0)
            events=hit_ground,  #  перестанет вычислять диффур на остальной области определения после того как событие произойдет
            dense_output=True,  # решение непрерывно
        )
        return result.y

    def visualise_data(self, data: ndarray[float]) -> None:
        """
        построит график на данных
        """
        #  axes - все элементы графика, figure - контейнер верхнего уровня для графика
        figure, axes = plt.subplots(figsize=(20, 20))  # размер побольше
        axes.plot(data[0][::10], data[1][::10])  # делаем оси x и y
        axes.grid()  # бьем на клетки

        axes.set_xlim(left=0)
        axes.set_ylim(bottom=0, top=20)
        axes.plot(data[0], data[1], color="red", linewidth=5, label="Траектория")

        plt.show()  # показать график


if __name__ == "__main__":
    Solver()
