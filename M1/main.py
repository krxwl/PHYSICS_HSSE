from numpy import ndarray # быстрый массивчик

from scipy.optimize import OptimizeResult
from scipy.integrate import solve_ivp  # обертка решения диффура
from scipy.constants import g

from math import cos, sin

import matplotlib
from constants import *


class Solver:
    def __init__(self):
        self.alpha, \
        self.v_0, \
        self.resistance_coefficient, \
        self.formula_choice_tag, \
        self.weight = self.read_data()


    def read_data(self) -> tuple[float, float, float, bool, float]:
        """
        считывает начальные условия
        """
        alpha: float = float(input(INPUT_ALPHA_STR))
        v_0: float = float(input(INPUT_INITIAL_SPEED_STR))
        resistance_coefficient: float = float(input(INPUT_COEFFICIENT_STR))

        formula_choice_flag: bool = bool(int(input(RESISTANCY_FORMULA_CHOICE_STR)))
        weight: float = float(input(INPUT_WEIGHT_STR))

        return tuple(alpha, v_0, resistance_coefficient, formula_choice_flag, weight)


    # coords - текущая точка в которой мы находимся
    def speed_equation(self, t: float, coordinates: ndarray[float]) -> list[float]:
        """
        решаем диффур для вязкого трения
        """
        x, y, v_x, v_y = coordinates

        # вычисляем производные в точке
        dvx_dt = - self.resistancy_coefficient / self.weight * v_x
        dvy_dt = -g - self.resistancy_coefficient / self.weight * v_y

        dx_dt = v_x
        dy_dt = v_y

        return ndarray([dx_dt, dy_dt, dvx_dt, dvy_dt])


    def hit_ground(self, t: float, coordinates: ndarray[float]) -> float:
        """
        функция которая когда тело упадет на землю (y=0) перестает
        интегрировать диффур на остальном интервале
        """
        return coordinates[1]


    def get_viscous_resistance_data(self) -> list[float]:
        # решаем задачу коши
        result: OptimizeResult = solve_ivp(fun=self.speed_equation, 
                                      t_span=(0, SPAN_MAX_UPPER_BORDER), # область интегрирования
                                      y0 = (0.0, 0.0, self.v_0 * cos(self.alpha), self.v_0 * sin(self.alpha)), # начальные условия (x0, y0, v_x0, v_y0)
                                      events=self.hit_ground,  #  перестанет вычислять диффур на остальной области определения после того как событие произойдет
                                      )
        


    def visualise_data(self, data: ndarray[float]) -> None:
        """
        построит график на данных
        """
        pass



if __name__ == "__main__":
    Solver()
