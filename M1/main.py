from numpy import ndarray, array, loadtxt, random  # быстрый массивчик

from scipy.integrate import solve_ivp  # обертка решения диффура
from scipy.constants import g

from math import cos, sin, radians, sqrt

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator  # для изменения масштаба

from constants import *

FORMULA_CHOICE_TAG = True

def random_rgb_numpy():
    return tuple(random.randint(0, 256, size=3))

# функция, показывающая, когда камень упадет
def hit_ground(t: float, coordinates: ndarray[float]) -> float:
    """
    Функция возвращает текущую высоту y.
    При падении тела на землю (y = 0) перестает интегрировать
    """
    return coordinates[1]


# Перестает интегрировать при срабатывании
hit_ground.terminal = True
# Срабатывает только при движении сверху вниз (то есть когда камень падает)
hit_ground.direction = -1


class Solver:
    def __init__(self):
        file = open('M1/data.txt', encoding='UTF-8')
        examples = loadtxt(file)
            
        """(
            self.alpha,
            self.v_0,
            self.resistance_coefficient,
            self.formula_choice_tag,
            self.weight,
        ) = self.read_data()"""
        results: ndarray[ndarray[float]] = []
        if FORMULA_CHOICE_TAG is False:
            self.current_model_name = "вязкое трение"
            for ex in examples:
                self.alpha = float(ex[0])
                self.v_0 = float(ex[1])
                self.resistance_coefficient = float(ex[2])
                self.weight = float(ex[3])
                results.append(self.get_viscous_resistance_data())
        else:
            self.current_model_name = "лобовое сопротивление"
            for ex in examples:
                self.alpha = float(ex[0])
                self.v_0 = float(ex[1])
                self.resistance_coefficient = float(ex[2])
                self.weight = float(ex[3])
                results.append(self.get_viscous_resistance_data())
        self.visualise_data(results)

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
    def speed_equation(self, t: float, coordinates: ndarray[float]) -> array[float]:
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

    def quad_speed_equation(self, t: float, coordinates: ndarray[float]) -> ndarray[float]:
        """
        решаем диффур для лобового сопротивления
        """
        x, y, v_x, v_y = coordinates
        v = sqrt(v_x**2 + v_y**2)

        # вычисляем производные в точке
        dvx_dt = -self.resistance_coefficient / self.weight * v_x * v
        dvy_dt = -g - self.resistance_coefficient / self.weight * v_y * v

        dx_dt = v_x
        dy_dt = v_y

        return ndarray([dx_dt, dy_dt, dvx_dt, dvy_dt])

    def get_start_conditions(self) -> list[float]:
        """
        функция возвращает начальное положение тела
        и проекции начальной скорости
        """
        x0 = 0.0
        y0 = 0.0001  # чтобы интегрирование не закончилось сразу же
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
            max_step=0.1
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
            max_step=0.1
        )
        return result.y

    def visualise_data(self, results: ndarray[ndarray[float]]) -> None:
        """
        построит график на данных
        """
        #  axes - все элементы графика, figure - контейнер верхнего уровня для графика
        figure, axes = plt.subplots(figsize=(20, 20))  # размер побольше
        # рисуем траекторию
        for data in results:
            axes.plot(
                data[0],
                data[1],
                color="red",
                linewidth=2.5,
                label=f"Траектория: {self.current_model_name}",
            )
        axes.grid(True, linestyle="--", alpha=0.6)  # бьем на клетки

        # делаем оси
        axes.set_xlim(left=0)
        axes.set_ylim(bottom=0, top=20)
        axes.set_xlabel("Дальность X, м", fontsize=11)
        axes.set_ylabel("Высота У, м", fontsize=11)
        axes.legend()

        plt.show()  # показать график


if __name__ == "__main__":
    Solver()
