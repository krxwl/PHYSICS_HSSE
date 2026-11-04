"""
physics.py — Физическое ядро и численное решение системы уравнений движения (Задача М4).
Разработано самостоятельно для моделирования динамики твёрдого тела.
"""

import numpy as np
from scipy.integrate import solve_ivp
from config import G, DT, RTOL, ATOL, SLIP_TOL, BODY_TYPES, validate_parameters


class RollingBodySimulator:
    """
    Класс для численного моделирования плоского движения симметричного тела
    (шар, мяч, цилиндр) по горизонтальной и наклонной шероховатой поверхности.
    Ось X направлена вдоль плоскости (при alpha > 0 — вниз по наклонной плоскости).
    Положительное направление вращения omega соответствует качению вдоль +X (v_c = v - omega * R).
    """

    def __init__(self, m, R, mu, alpha_deg, body_key="solid_sphere"):
        if body_key not in BODY_TYPES:
            raise ValueError(f"Неизвестный тип тела: {body_key}")
        self.body_info = BODY_TYPES[body_key]
        self.beta = self.body_info["beta"]
        validate_parameters(m, R, mu, alpha_deg, self.beta)

        self.m = float(m)
        self.R = float(R)
        self.mu = float(mu)
        self.alpha_deg = float(alpha_deg)
        self.alpha = np.radians(alpha_deg)
        self.I = self.beta * self.m * (self.R ** 2)  # Момент инерции относительно оси через ЦМ

        # Сила нормальной реакции опоры и максимальная сила трения покоя/скольжения
        self.N = self.m * G * np.cos(self.alpha)
        self.F_fr_max = self.mu * self.N

        # Критический коэффициент трения для чистого качения под действием тяжести
        # mu_crit = (beta / (1 + beta)) * tan(alpha)
        self.mu_crit = (self.beta / (1.0 + self.beta)) * np.tan(self.alpha)

    def can_maintain_pure_rolling(self):
        """
        Проверяет, способна ли сила трения покоя удерживать чистое качение без проскальзывания.
        Условие: F_tr_static = m*g*sin(alpha) * beta / (1 + beta) <= mu * m*g*cos(alpha).
        """
        return self.mu >= self.mu_crit - 1e-12

    def compute_forces_and_derivatives(self, state, mode):
        """
        Вычисляет правые части системы ОДУ [dx/dt, dv/dt, dphi/dt, domega/dt, dA_fr/dt]
        и текущую силу трения F_fr (проекция на ось +X).
        Вектор состояния state = [x, v, phi, omega, A_fr_loss],
        где A_fr_loss >= 0 — накопленная диссипированная работа силы трения.
        """
        x, v, phi, omega, _ = state
        v_c = v - omega * self.R  # Скорость нижней точки контакта

        if mode == "rolling":
            # Режим качения без проскальзывания: v = omega * R => a = eps * R
            # Из m*a = m*g*sin(alpha) + F_fr и I*eps = -F_fr * R получаем:
            a = (G * np.sin(self.alpha)) / (1.0 + self.beta)
            eps = a / self.R
            F_fr = -self.m * G * np.sin(self.alpha) * (self.beta / (1.0 + self.beta))
            d_loss = 0.0  # В режиме чистого качения мощность силы трения покоя равна нулю
        else:
            # Режим качения с проскальзыванием (сухое трение Кулона)
            # Сила трения направлена строго против скорости точки контакта v_c
            sign_vc = 1.0 if v_c >= 0.0 else -1.0
            F_fr = -sign_vc * self.F_fr_max
            a = G * np.sin(self.alpha) + F_fr / self.m
            eps = (-F_fr * self.R) / self.I
            # Мощность диссипации механической энергии за счёт трения скольжения: P_loss = |F_fr * v_c|
            d_loss = abs(F_fr * v_c)

        return np.array([v, a, omega, eps, d_loss]), F_fr

    def simulate(self, x0, v0, omega0, t_max):
        """
        Численно интегрирует уравнения движения на отрезке [0, t_max]
        с автоматическим определением момента смены режима (скольжение <-> чистое качение).
        """
        v_c0 = v0 - omega0 * self.R
        # Определяем начальный режим движения
        if abs(v_c0) <= SLIP_TOL and self.can_maintain_pure_rolling():
            current_mode = "rolling"
        else:
            current_mode = "slipping"

        state0 = np.array([float(x0), float(v0), 0.0, float(omega0), 0.0])
        t_current = 0.0

        t_all = []
        state_all = []
        mode_all = []
        t_transition = None

        # Цикл по фазам движения (до и после смены режима)
        while t_current < t_max - 1e-12:
            t_eval = np.arange(t_current, t_max, DT)
            if len(t_eval) == 0 or t_eval[-1] < t_max:
                t_eval = np.append(t_eval, t_max)

            def ode_rhs(t, y):
                derivs, _ = self.compute_forces_and_derivatives(y, current_mode)
                return derivs

            if current_mode == "slipping":
                # Событие обнуления скорости точки контакта: v - omega * R = 0
                def slip_zero_event(t, y):
                    return y[1] - y[3] * self.R

                slip_zero_event.terminal = True
                slip_zero_event.direction = 0

                sol = solve_ivp(
                    ode_rhs,
                    (t_current, t_max),
                    state0,
                    method="RK45",
                    t_eval=t_eval,
                    events=slip_zero_event,
                    rtol=RTOL,
                    atol=ATOL,
                )
            else:
                sol = solve_ivp(
                    ode_rhs,
                    (t_current, t_max),
                    state0,
                    method="RK45",
                    t_eval=t_eval,
                    rtol=RTOL,
                    atol=ATOL,
                )

            # Сохраняем рассчитанный участок траектории
            if len(t_all) > 0:
                t_all.append(sol.t[1:])
                state_all.append(sol.y[:, 1:])
                mode_all.extend([current_mode] * (len(sol.t) - 1))
            else:
                t_all.append(sol.t)
                state_all.append(sol.y)
                mode_all.extend([current_mode] * len(sol.t))

            t_current = sol.t[-1]
            state0 = sol.y[:, -1].copy()

            # Если произошло событие остановки проскальзывания
            if current_mode == "slipping" and sol.status == 1 and len(sol.t_events[0]) > 0:
                t_transition = float(sol.t_events[0][0])
                # Выравниваем скорость и угловую скорость точно по условию v = omega * R
                state0[1] = state0[3] * self.R
                if self.can_maintain_pure_rolling():
                    current_mode = "rolling"
                else:
                    # Если угол слишком крутой (mu < mu_crit), тело снова начнёт скользить вниз
                    current_mode = "slipping"

        t_arr = np.concatenate(t_all)
        y_arr = np.hstack(state_all)

        return self._postprocess_results(t_arr, y_arr, mode_all, t_transition)

    def _postprocess_results(self, t, y, modes, t_transition):
        """
        Формирует полный набор физических величин, проверяет законы сохранения/изменения
        энергии и момента импульса.
        """
        x = y[0]
        v = y[1]
        phi = y[2]
        omega = y[3]
        A_fr_loss = y[4]  # Величина потерь энергии на трение (>= 0), работа трения A_fr = -A_fr_loss

        v_c = v - omega * self.R

        # Ускорения и сила трения в каждый момент времени
        a = np.zeros_like(t)
        eps = np.zeros_like(t)
        F_fr = np.zeros_like(t)
        for i in range(len(t)):
            derivs, f_val = self.compute_forces_and_derivatives(y[:, i], modes[i])
            a[i] = derivs[1]
            eps[i] = derivs[3]
            F_fr[i] = f_val

        # Энергетические характеристики
        # Нулевой уровень потенциальной энергии выбран в точке x = 0: U(x) = -m * g * x * sin(alpha)
        E_trans = 0.5 * self.m * (v ** 2)
        E_rot = 0.5 * self.I * (omega ** 2)
        E_kin = E_trans + E_rot
        U_pot = -self.m * G * x * np.sin(self.alpha)
        E_mech = E_kin + U_pot

        # Закон изменения механической энергии: E_mech(t) - E_mech(0) = A_fr = -A_fr_loss
        # Следовательно, инвариант E_total_conserved = E_mech(t) + A_fr_loss(t) должен быть равен E_mech(0)
        E_total_conserved = E_mech + A_fr_loss
        energy_error = np.max(np.abs(E_total_conserved - E_mech[0]))

        # Момент импульса относительно неподвижной оси на плоскости (точка контакта в t=0):
        # Для плоского движения L_O = m * v * R + I * omega
        # Момент силы трения и нормальной реакции относительно точки на плоскости равен нулю!
        # Момент силы тяжести относительно линии контакта равен M_g = m * g * sin(alpha) * R.
        # Поэтому dL_O/dt = m * g * R * sin(alpha) => L_O(t) - m*g*R*sin(alpha)*t = const = L_O(0)!
        L_O = self.m * v * self.R + self.I * omega
        L_O_invariant = L_O - self.m * G * self.R * np.sin(self.alpha) * t
        angular_momentum_error = np.max(np.abs(L_O_invariant - L_O[0]))

        return {
            "t": t,
            "x": x,
            "v": v,
            "a": a,
            "phi": phi,
            "omega": omega,
            "eps": eps,
            "v_c": v_c,
            "F_fr": F_fr,
            "modes": modes,
            "t_transition": t_transition,
            "E_trans": E_trans,
            "E_rot": E_rot,
            "E_kin": E_kin,
            "U_pot": U_pot,
            "E_mech": E_mech,
            "A_fr_loss": A_fr_loss,
            "E_total_conserved": E_total_conserved,
            "energy_error": energy_error,
            "L_O": L_O,
            "L_O_invariant": L_O_invariant,
            "angular_momentum_error": angular_momentum_error,
        }

    def analytical_solution(self, t, x0, v0, omega0):
        """
        Аналитическое решение для движения тела по горизонтальной или наклонной плоскости
        с начальным проскальзыванием и последующим переходом в чистое качение (или без перехода).
        Используется для верификации численной модели.
        """
        v_c0 = v0 - omega0 * self.R
        beta = self.beta

        # Если с самого начала чистое качение
        if abs(v_c0) <= SLIP_TOL and self.can_maintain_pure_rolling():
            t_star = 0.0
            a_roll = G * np.sin(self.alpha) / (1.0 + beta)
            v_th = v0 + a_roll * t
            omega_th = v_th / self.R
            x_th = x0 + v0 * t + 0.5 * a_roll * (t ** 2)
            return {
                "t_star": t_star,
                "x": x_th,
                "v": v_th,
                "omega": omega_th,
                "v_final_horiz": v0 if self.alpha == 0 else None,
            }

        # Направление начального скольжения s = sign(v_c0)
        s = 1.0 if v_c0 >= 0.0 else -1.0
        a_slip = G * np.sin(self.alpha) - s * self.mu * G * np.cos(self.alpha)
        eps_slip = (s * self.mu * G * np.cos(self.alpha)) / (beta * self.R)

        # Скорость точки контакта меняется по закону: v_c(t) = v_c0 + (a_slip - eps_slip * R) * t
        rate_vc = a_slip - eps_slip * self.R

        # Проверяем, уменьшается ли |v_c(t)| со временем (т.е. имеет ли t_star > 0)
        if s * rate_vc < 0:
            t_star = -v_c0 / rate_vc
        else:
            t_star = np.inf  # Проскальзывание никогда не прекратится (например, при крутом угле и малом mu)

        x_th = np.zeros_like(t)
        v_th = np.zeros_like(t)
        omega_th = np.zeros_like(t)

        for i, ti in enumerate(t):
            if ti <= t_star or not self.can_maintain_pure_rolling():
                v_th[i] = v0 + a_slip * ti
                omega_th[i] = omega0 + eps_slip * ti
                x_th[i] = x0 + v0 * ti + 0.5 * a_slip * (ti ** 2)
            else:
                # Состояние в момент перехода t_star
                v_star = v0 + a_slip * t_star
                omega_star = omega0 + eps_slip * t_star
                x_star = x0 + v0 * t_star + 0.5 * a_slip * (t_star ** 2)

                dt_roll = ti - t_star
                a_roll = G * np.sin(self.alpha) / (1.0 + beta)
                v_th[i] = v_star + a_roll * dt_roll
                omega_th[i] = v_th[i] / self.R
                x_th[i] = x_star + v_star * dt_roll + 0.5 * a_roll * (dt_roll ** 2)

        # Теоретическая установившаяся скорость для горизонтальной плоскости из ЗСМИ:
        # L_O = m*v0*R + I*omega0 = (m*R + I/R)*v_ust => v_ust = (v0 + beta*R*omega0) / (1 + beta)
        v_ust_horiz = (v0 + beta * self.R * omega0) / (1.0 + beta) if self.alpha == 0 else None

        return {
            "t_star": t_star if t_star != np.inf else None,
            "x": x_th,
            "v": v_th,
            "omega": omega_th,
            "v_final_horiz": v_ust_horiz,
        }