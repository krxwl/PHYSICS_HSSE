"""
physics.py — Физическое ядро и численное решение системы уравнений движения
Решатель обрабатывает скольжение, смену его направления и чистое качение
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
        self.I = self.beta * self.m * (self.R**2)

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
            sign_vc = (
                fixed_sign_vc
                if fixed_sign_vc is not None
                else (1.0 if v_c >= 0.0 else -1.0)
            )
            F_fr = -sign_vc * self.F_fr_max
            a = G * np.sin(self.alpha) + F_fr / self.m
            eps = (-F_fr * self.R) / self.I
            d_loss = abs(F_fr * v_c)

        return np.array([v, a, omega, eps, d_loss]), F_fr

    def _build_time_grid(self, t_max):
        """Создаёт сетку моментов, в которых нужно сохранить решение."""
        t_grid = np.arange(0.0, t_max, DT)
        if len(t_grid) == 0 or t_grid[-1] < t_max:
            t_grid = np.append(t_grid, t_max)
        return t_grid

    def _integrate_segment(self, mode, state, t_start, t_max, t_eval, slip_sign):
        """Интегрирует один участок движения с постоянным режимом трения."""

        def right_hand_side(t, values):
            derivs, _ = self.compute_forces_and_derivatives(
                values, mode, slip_sign if mode == "slipping" else None
            )
            return derivs

        events = None
        if mode == "slipping":

            def slip_zero_event(t, values):
                return values[1] - values[3] * self.R

            slip_zero_event.terminal = True
            slip_zero_event.direction = -slip_sign
            events = slip_zero_event

        solution = solve_ivp(
            right_hand_side,
            (t_start, t_max),
            state,
            method="RK45",
            t_eval=t_eval,
            events=events,
            rtol=RTOL,
            atol=ATOL,
        )
        if not solution.success:
            raise RuntimeError(f"Не удалось интегрировать движение: {solution.message}")
        return solution

    def _segment_results(self, solution, t_start, state, mode, can_roll):
        """Подготавливает массивы одного участка и при необходимости добавляет событие."""
        segment_t = np.concatenate(([t_start], solution.t))
        segment_state = np.column_stack((state, solution.y))
        segment_modes = [mode] * len(segment_t)
        event_found = (
            mode == "slipping"
            and solution.t_events is not None
            and len(solution.t_events[0]) > 0
        )
        if not event_found:
            return segment_t, segment_state, segment_modes, None, None

        t_event = float(solution.t_events[0][0])
        state_event = solution.y_events[0][0].copy()
        # Убираем остаточную погрешность из условия нулевой скорости контакта.
        state_event[1] = state_event[3] * self.R

        if abs(segment_t[-1] - t_event) <= 1e-12:
            segment_t[-1] = t_event
            segment_state[:, -1] = state_event
        else:
            segment_t = np.append(segment_t, t_event)
            segment_state = np.column_stack((segment_state, state_event))
            segment_modes.append(mode)

        if can_roll:
            # В точке перехода используем силу трения покоя.
            segment_modes[-1] = "rolling"
        return segment_t, segment_state, segment_modes, t_event, state_event

    @staticmethod
    def _append_segment(
        t_parts, state_parts, mode_all, segment_t, segment_state, modes
    ):
        """Добавляет участок, не дублируя общую граничную точку."""
        if t_parts:
            t_parts.append(segment_t[1:])
            state_parts.append(segment_state[:, 1:])
            mode_all.extend(modes[1:])
        else:
            t_parts.append(segment_t)
            state_parts.append(segment_state)
            mode_all.extend(modes)

    def simulate(self, x0, v0, omega0, t_max):
        """Считает движение по участкам через solve_ivp и объединяет результаты."""
        if not np.isfinite([x0, v0, omega0, t_max]).all():
            raise ValueError("Начальные условия и t_max должны быть конечными числами.")
        if t_max <= 0.0:
            raise ValueError("Время моделирования t_max должно быть положительным.")

        v_c0 = v0 - omega0 * self.R
        can_roll = self.can_maintain_pure_rolling()
        if abs(v_c0) <= SLIP_TOL and can_roll:
            mode, slip_sign = "rolling", None
        else:
            mode = "slipping"
            # При старте из покоя недостаточное трение задаёт скольжение вниз по склону.
            slip_sign = 1.0 if abs(v_c0) <= SLIP_TOL or v_c0 > 0.0 else -1.0

        t_grid = self._build_time_grid(t_max)
        t_parts, state_parts, mode_all = [], [], []
        state = np.array([float(x0), float(v0), 0.0, float(omega0), 0.0])
        t_current = 0.0
        t_transition = None
        slip_reversals = 0

        while t_current < t_max - 1e-14:
            t_eval = t_grid[t_grid > t_current + 1e-14]
            solution = self._integrate_segment(
                mode, state, t_current, t_max, t_eval, slip_sign
            )
            segment = self._segment_results(solution, t_current, state, mode, can_roll)
            segment_t, segment_state, segment_modes, t_event, state_event = segment
            self._append_segment(
                t_parts, state_parts, mode_all, segment_t, segment_state, segment_modes
            )

            if t_event is None:
                break

            t_current, state = t_event, state_event
            if can_roll:
                t_transition = t_event
                mode = "rolling"
            else:
                # Если сцепления недостаточно, после события скольжение меняет знак.
                slip_sign *= -1.0
                slip_reversals += 1
                if slip_reversals > 2:
                    raise RuntimeError(
                        "Слишком много смен знака скорости проскальзывания."
                    )

        t_arr = np.concatenate(t_parts)
        y_arr = np.hstack(state_parts)
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

        E_trans = 0.5 * self.m * (v**2)
        E_rot = 0.5 * self.I * (omega**2)
        E_kin = E_trans + E_rot
        U_pot = -self.m * G * x * np.sin(self.alpha)
        E_mech = E_kin + U_pot

        E_total_conserved = E_mech + A_fr_loss
        energy_error = np.max(np.abs(E_total_conserved - E_mech[0]))

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

    def _slipping_accelerations(self, slip_sign):
        """Возвращает линейное, угловое ускорения и скорость изменения v_c."""
        a = G * np.sin(self.alpha) - slip_sign * self.mu * G * np.cos(self.alpha)
        eps = slip_sign * self.mu * G * np.cos(self.alpha) / (self.beta * self.R)
        return a, eps, a - eps * self.R

    def _slipping_state_at(self, t, t_start, state_start, slip_sign):
        """Вычисляет x, v и omega на участке скольжения с постоянным знаком трения."""
        x_start, v_start, omega_start = state_start
        a, eps, _ = self._slipping_accelerations(slip_sign)
        dt = np.asarray(t, dtype=float) - t_start
        x = x_start + v_start * dt + 0.5 * a * dt**2
        v = v_start + a * dt
        omega = omega_start + eps * dt
        return x, v, omega

    def _rolling_state_at(self, t, t_start, x_start, v_start):
        """Вычисляет координату и скорости при чистом качении."""
        dt = np.asarray(t, dtype=float) - t_start
        a_roll = G * np.sin(self.alpha) / (1.0 + self.beta)
        v = v_start + a_roll * dt
        x = x_start + v_start * dt + 0.5 * a_roll * dt**2
        omega = v / self.R
        return x, v, omega

    def _slip_event_time(self, t_start, state_start, slip_sign, t_max):
        """Вычисляет момент обращения v_c в ноль, если он попадает в интервал."""
        _, _, rate_vc = self._slipping_accelerations(slip_sign)
        v_c_start = state_start[1] - state_start[2] * self.R
        if abs(v_c_start) <= SLIP_TOL or slip_sign * rate_vc >= 0.0:
            return None

        t_event = t_start - v_c_start / rate_vc
        if t_start <= t_event <= t_max:
            return t_event
        return None

    def _analytical_slipping_solution(self, t, x0, v0, omega0, can_roll):
        """Собирает аналитическое решение из участков скольжения и, при возможности, качения."""
        x_th = np.empty_like(t)
        v_th = np.empty_like(t)
        omega_th = np.empty_like(t)
        remaining = np.ones(t.shape, dtype=bool)
        t_start = 0.0
        state_start = (float(x0), float(v0), float(omega0))
        v_c0 = v0 - omega0 * self.R
        slip_sign = 1.0 if abs(v_c0) <= SLIP_TOL or v_c0 > 0.0 else -1.0
        t_max = float(np.max(t)) if t.size else 0.0
        t_star = None

        # После разворота скорость контакта уходит от нуля.
        for _ in range(3):
            t_event = self._slip_event_time(t_start, state_start, slip_sign, t_max)
            if t_event is None:
                segment_mask = remaining
            else:
                segment_mask = remaining & (t <= t_event + 1e-12)

            states = self._slipping_state_at(
                t[segment_mask], t_start, state_start, slip_sign
            )
            x_th[segment_mask], v_th[segment_mask], omega_th[segment_mask] = states
            remaining[segment_mask] = False

            if t_event is None:
                break

            event_state = self._slipping_state_at(
                np.array([t_event]), t_start, state_start, slip_sign
            )
            event_x = float(event_state[0][0])
            event_omega = float(event_state[2][0])
            event_v = event_omega * self.R

            if can_roll:
                t_star = t_event
                rolling_states = self._rolling_state_at(
                    t[remaining], t_event, event_x, event_v
                )
                x_th[remaining], v_th[remaining], omega_th[remaining] = rolling_states
                remaining[:] = False
                break

            # Трения покоя недостаточно: на следующем участке знак трения меняется.
            t_start = t_event
            state_start = (event_x, event_v, event_omega)
            slip_sign *= -1.0

        if np.any(remaining):
            raise RuntimeError(
                "Не удалось построить аналитическое решение для всех моментов времени."
            )
        return x_th, v_th, omega_th, t_star

    def analytical_solution(self, t, x0, v0, omega0):
        """Строит аналитические x(t), v(t), omega(t) для сравнения с интегратором."""
        t_arr = np.asarray(t, dtype=float)
        if t_arr.ndim == 0:
            t_arr = t_arr.reshape(1)

        v_c0 = v0 - omega0 * self.R
        can_roll = self.can_maintain_pure_rolling()
        if abs(v_c0) <= SLIP_TOL and can_roll:
            x_th, v_th, omega_th = self._rolling_state_at(t_arr, 0.0, x0, v0)
            t_star = 0.0
        else:
            x_th, v_th, omega_th, t_star = self._analytical_slipping_solution(
                t_arr, x0, v0, omega0, can_roll
            )

        v_ust_horiz = (
            (v0 + self.beta * self.R * omega0) / (1.0 + self.beta)
            if self.alpha == 0
            else None
        )
        return {
            "t_star": t_star,
            "x": x_th,
            "v": v_th,
            "omega": omega_th,
            "v_final_horiz": v_ust_horiz,
        }
