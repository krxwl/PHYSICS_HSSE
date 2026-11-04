"""
physics.py — Физическое ядро и численное решение системы уравнений движения (Задача М4).
Стабильная двухфазная схема численного интегрирования (RK45).
"""

import numpy as np
from scipy.integrate import solve_ivp
from config import G, DT, RTOL, ATOL, SLIP_TOL, BODY_TYPES, validate_parameters


class RollingBodySimulator:
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
        self.I = self.beta * self.m * (self.R ** 2)

        self.N = self.m * G * np.cos(self.alpha)
        self.F_fr_max = self.mu * self.N
        self.mu_crit = (self.beta / (1.0 + self.beta)) * np.tan(self.alpha)

    def can_maintain_pure_rolling(self):
        return self.mu >= self.mu_crit - 1e-12

    def compute_forces_and_derivatives(self, state, mode, fixed_sign_vc=None):
        x, v, phi, omega, _ = state
        v_c = v - omega * self.R

        if mode == "rolling":
            a = (G * np.sin(self.alpha)) / (1.0 + self.beta)
            eps = a / self.R
            F_fr = -self.m * G * np.sin(self.alpha) * (self.beta / (1.0 + self.beta))
            d_loss = 0.0
        else:
            sign_vc = fixed_sign_vc if fixed_sign_vc is not None else (1.0 if v_c >= 0.0 else -1.0)
            F_fr = -sign_vc * self.F_fr_max
            a = G * np.sin(self.alpha) + F_fr / self.m
            eps = (-F_fr * self.R) / self.I
            d_loss = abs(F_fr * v_c)

        return np.array([v, a, omega, eps, d_loss]), F_fr

    def simulate(self, x0, v0, omega0, t_max):
        """
        Численно интегрирует ОДУ через solve_ivp (RK45).
        Гарантирует отсутствие зависаний за счёт разбиения на фазы скольжения и качения.
        """
        v_c0 = v0 - omega0 * self.R
        s0 = 1.0 if v_c0 >= 0.0 else -1.0

        a_slip = G * np.sin(self.alpha) - s0 * self.mu * G * np.cos(self.alpha)
        eps_slip = (s0 * self.mu * G * np.cos(self.alpha)) / (self.beta * self.R)
        rate_vc = a_slip - eps_slip * self.R

        can_roll = self.can_maintain_pure_rolling()

        # Расчет времени смены режима
        if abs(v_c0) <= SLIP_TOL and can_roll:
            has_transition = False
            t_star = 0.0
            initial_mode = "rolling"
        elif s0 * rate_vc < 0 and can_roll:
            t_star = -v_c0 / rate_vc
            has_transition = (0.0 < t_star < t_max)
            initial_mode = "slipping"
        else:
            has_transition = False
            t_star = None
            initial_mode = "slipping"

        t_all = []
        state_all = []
        mode_all = []
        t_transition = t_star if (t_star is not None and 0.0 < t_star < t_max) else None

        if not has_transition:
            # Один режим на всём интервале [0, t_max]
            t_eval = np.arange(0.0, t_max, DT)
            if t_eval[-1] < t_max:
                t_eval = np.append(t_eval, t_max)

            def ode_one(t, y):
                derivs, _ = self.compute_forces_and_derivatives(y, initial_mode, s0)
                return derivs

            y0 = np.array([float(x0), float(v0), 0.0, float(omega0), 0.0])
            sol = solve_ivp(ode_one, (0.0, t_max), y0, method="RK45", t_eval=t_eval, rtol=RTOL, atol=ATOL)
            t_arr = sol.t
            y_arr = sol.y
            mode_all = [initial_mode] * len(sol.t)
        else:
            # Фаза 1: скольжение от 0 до t_star
            t_eval1 = np.arange(0.0, t_star, DT)
            if len(t_eval1) == 0 or t_eval1[-1] < t_star:
                t_eval1 = np.append(t_eval1, t_star)

            def ode_slip(t, y):
                derivs, _ = self.compute_forces_and_derivatives(y, "slipping", s0)
                return derivs

            y0 = np.array([float(x0), float(v0), 0.0, float(omega0), 0.0])
            sol1 = solve_ivp(ode_slip, (0.0, t_star), y0, method="RK45", t_eval=t_eval1, rtol=RTOL, atol=ATOL)

            t_all.append(sol1.t)
            state_all.append(sol1.y)
            mode_all.extend(["slipping"] * len(sol1.t))

            # Сшивка в точке перехода t_star: v = omega * R
            state_trans = sol1.y[:, -1].copy()
            state_trans[1] = state_trans[3] * self.R

            # Фаза 2: чистое качение от t_star до t_max
            t_eval2 = np.arange(t_star, t_max, DT)
            if len(t_eval2) == 0 or t_eval2[-1] < t_max:
                t_eval2 = np.append(t_eval2, t_max)

            def ode_roll(t, y):
                derivs, _ = self.compute_forces_and_derivatives(y, "rolling")
                return derivs

            sol2 = solve_ivp(ode_roll, (t_star, t_max), state_trans, method="RK45", t_eval=t_eval2, rtol=RTOL, atol=ATOL)

            # Объединение без дублирования точки t_star
            t_all.append(sol2.t[1:])
            state_all.append(sol2.y[:, 1:])
            mode_all.extend(["rolling"] * (len(sol2.t) - 1))

            t_arr = np.concatenate(t_all)
            y_arr = np.hstack(state_all)

        return self._postprocess_results(t_arr, y_arr, mode_all, t_transition)

    def _postprocess_results(self, t, y, modes, t_transition):
        x, v, phi, omega, A_fr_loss = y[0], y[1], y[2], y[3], y[4]
        v_c = v - omega * self.R

        a = np.zeros_like(t)
        eps = np.zeros_like(t)
        F_fr = np.zeros_like(t)
        for i in range(len(t)):
            derivs, f_val = self.compute_forces_and_derivatives(y[:, i], modes[i])
            a[i] = derivs[1]
            eps[i] = derivs[3]
            F_fr[i] = f_val

        E_trans = 0.5 * self.m * (v ** 2)
        E_rot = 0.5 * self.I * (omega ** 2)
        E_kin = E_trans + E_rot
        U_pot = -self.m * G * x * np.sin(self.alpha)
        E_mech = E_kin + U_pot

        E_total_conserved = E_mech + A_fr_loss
        energy_error = np.max(np.abs(E_total_conserved - E_mech[0]))

        L_O = self.m * v * self.R + self.I * omega
        L_O_invariant = L_O - self.m * G * self.R * np.sin(self.alpha) * t
        angular_momentum_error = np.max(np.abs(L_O_invariant - L_O[0]))

        return {
            "t": t, "x": x, "v": v, "a": a, "phi": phi, "omega": omega, "eps": eps,
            "v_c": v_c, "F_fr": F_fr, "modes": modes, "t_transition": t_transition,
            "E_trans": E_trans, "E_rot": E_rot, "E_kin": E_kin, "U_pot": U_pot,
            "E_mech": E_mech, "A_fr_loss": A_fr_loss, "E_total_conserved": E_total_conserved,
            "energy_error": energy_error, "L_O": L_O, "L_O_invariant": L_O_invariant,
            "angular_momentum_error": angular_momentum_error,
        }

    def analytical_solution(self, t, x0, v0, omega0):
        v_c0 = v0 - omega0 * self.R
        beta = self.beta

        if abs(v_c0) <= SLIP_TOL and self.can_maintain_pure_rolling():
            t_star = 0.0
            a_roll = G * np.sin(self.alpha) / (1.0 + beta)
            v_th = v0 + a_roll * t
            omega_th = v_th / self.R
            x_th = x0 + v0 * t + 0.5 * a_roll * (t ** 2)
            return {"t_star": t_star, "x": x_th, "v": v_th, "omega": omega_th, "v_final_horiz": v0 if self.alpha == 0 else None}

        s = 1.0 if v_c0 >= 0.0 else -1.0
        a_slip = G * np.sin(self.alpha) - s * self.mu * G * np.cos(self.alpha)
        eps_slip = (s * self.mu * G * np.cos(self.alpha)) / (beta * self.R)
        rate_vc = a_slip - eps_slip * self.R

        if s * rate_vc < 0:
            t_star = -v_c0 / rate_vc
        else:
            t_star = np.inf

        x_th = np.zeros_like(t)
        v_th = np.zeros_like(t)
        omega_th = np.zeros_like(t)

        for i, ti in enumerate(t):
            if ti <= t_star or not self.can_maintain_pure_rolling():
                v_th[i] = v0 + a_slip * ti
                omega_th[i] = omega0 + eps_slip * ti
                x_th[i] = x0 + v0 * ti + 0.5 * a_slip * (ti ** 2)
            else:
                v_star = v0 + a_slip * t_star
                x_star = x0 + v0 * t_star + 0.5 * a_slip * (t_star ** 2)
                dt_roll = ti - t_star
                a_roll = G * np.sin(self.alpha) / (1.0 + beta)
                v_th[i] = v_star + a_roll * dt_roll
                omega_th[i] = v_th[i] / self.R
                x_th[i] = x_star + v_star * dt_roll + 0.5 * a_roll * (dt_roll ** 2)

        v_ust_horiz = (v0 + beta * self.R * omega0) / (1.0 + beta) if self.alpha == 0 else None

        return {
            "t_star": t_star if t_star != np.inf else None,
            "x": x_th, "v": v_th, "omega": omega_th, "v_final_horiz": v_ust_horiz,
        }