from numpy import ndarray # быстрый массивчик

from scipy import solve_ivp  # для решения диффуров
from scipy.integrate import OdeResult  # обертка решения диффура
from scipy.constants import g

from math import cos, sin

import matplotlib
from constants import *


def read_data() -> tuple[float, float, float, bool]:
    """
    считывает начальные условия
    """
    alpha: float = float(input(INPUT_ALPHA_STR))
    v_0: float = float(input(INPUT_INITIAL_SPEED_STR))
    resistance_coefficient: float = float(input(INPUT_COEFFICIENT))

    formula_choice_flag: bool = bool(int(input()))
    return tuple(alpha, v_0, resistance_coefficient, formula_choice_flag)


# coords - все неизвестные величины
def speed_equation(t: float, coordinates: ndarray[float]) -> list[float]:
    """
    решаем диффур для вязкого трения
    """
    x, y, v_x, v_x = coordinates
    # TODO( ОБЪЕДИНИТЬ В КЛАСС ДЛЯ ВИДИМОСТИ НАЧАЛЬНЫХ УСЛОВИЙ)
    # зависимости величин
    dvx_dt = - resistancy_coefficient / weight * v_x
    dvy_dt = -g - resistancy_coefficient / m * v_y

    dx_dt = v_x
    dy_dt = v_y

    # возвращаем все "скорости величин"
    return ndarray([dx_dt, dy_dt, dvx_dt, dvy_dt])


def hit_ground(t: float, coordinates: ndarray[float]) -> float:
    """
    функция которая когда скорость становится равна нулю перестает
    вычислять диффур на остальной области определения
    """
    return coordinates[1]  # если y нулевой то остановимся


def get_viscous_resistance_data() -> list[float]:
    # решаем задачу коши
    # TODO( ОБЪЕДИНИТЬ В КЛАСС ДЛЯ ВИДИМОСТИ НАЧАЛЬНЫХ УСЛОВИЙ)
    result: OdeResult = solve_ivp(fun=speed_equation,
                                  t_span=(0, SPAN_MAX_UPPER_BORDER),
                                  y0 = (0.0, 0.0, v_0 * cos(alpha), v_0 * sin(alpha)), # начальные условия (x0, y0, v_x0, v_y0)
                                  events=hit_ground,  #  перестанет вычислять диффур на остальной области определения после того как событие произойдет
                                  method='RK45')


def visualise_data(data: ndarray[float]) -> None:
    """
    построит график на данных
    """
    pass


def run_program() -> None:
    while True:
        # TODO( ОБЪЕДИНИТЬ В КЛАСС ДЛЯ ВИДИМОСТИ НАЧАЛЬНЫХ УСЛОВИЙ В ДРУГИХ ФУНКЦИЯХ)
        alpha, v_0, resistance_coefficient, formula_choice_flag = read_data()

        if formula_choice_flag is True:
            pass
            # ветка для вязкого трения
        else:
            pass
            # ветка для лобового сопротивления


if __name__ == "__main__":
    run_program()
