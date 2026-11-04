"""
Основной исполняемый файл
Запускает серию численных экспериментов, сравнивает результаты с аналитическими
решениями, проверяет выполнение законов сохранения и выводит графики.
"""

from pathlib import Path

from config import SCENARIOS
from physics import RollingBodySimulator
from visualization import plot_single_experiment, plot_incline_comparison


def print_experiment_summary(name, sim, res, theory):
    """Выводит в консоль результаты расчёта и сравнение с теорией."""
    print("=" * 85)
    print(f" {name}")
    print("=" * 85)
    print(
        f"Тело: {sim.body_info['name']} (beta = {sim.beta:.4f}), m = {sim.m} кг, R = {sim.R} м"
    )
    print(
        f"Параметры плоскости: угол alpha = {sim.alpha_deg}°, коэф. трения mu = {sim.mu}"
    )
    if sim.alpha_deg > 0:
        print(
            f"Критический коэф. трения для чистого качения: mu_crit = {sim.mu_crit:.4f}"
        )

    # Сравнение времени перехода к качению без проскальзывания
    t_num = res["t_transition"]
    t_th = theory["t_star"]
    if t_num is not None and t_th is not None:
        err_t = abs(t_num - t_th)
        print("Момент смены режима t* (скольжение -> качение):")
        print(f"  Численно:     t* = {t_num:.6f} с")
        print(f"  Теоретически: t* = {t_th:.6f} с  | Абс. погрешность: {err_t:.2e} с")
    else:
        print(
            "Режим движения не менялся (либо чистое качение с t=0, либо непрерывное скольжение)."
        )

    # Сравнение конечной скорости и координаты
    v_end_num, v_end_th = res["v"][-1], theory["v"][-1]
    x_end_num, x_end_th = res["x"][-1], theory["x"][-1]
    print("Скорость ЦМ в конце расчёта v(t_max):")
    print(f"  Численно:     v = {v_end_num:.6f} м/с")
    print(
        f"  Теоретически: v = {v_end_th:.6f} м/с | Погрешность: {abs(v_end_num - v_end_th):.2e} м/с"
    )
    print("Координата ЦМ в конце расчёта x(t_max):")
    print(f"  Численно:     x = {x_end_num:.6f} м")
    print(
        f"  Теоретически: x = {x_end_th:.6f} м   | Погрешность: {abs(x_end_num - x_end_th):.2e} м"
    )

    # Проверка аналитической формулы через ЗСМИ для горизонтального стола: v_уст = (v0 + beta*R*omega0)/(1+beta)
    if theory["v_final_horiz"] is not None:
        E0 = res["E_mech"][0]
        Ef = res["E_mech"][-1]
        loss_ratio_num = (E0 - Ef) / E0 if E0 > 0 else 0.0
        # Для удара без вращения (omega0 = 0) доля потерь энергии равна beta / (1 + beta)
        print(
            f"Установившаяся скорость на горизонт. столе из ЗСМИ: v_уст = {theory['v_final_horiz']:.6f} м/с"
        )
        if abs(res["omega"][0]) < 1e-9:
            loss_ratio_th = sim.beta / (1.0 + sim.beta)
            print("Доля диссипированной энергии Delta E / E0:")
            print(f"  Численно:     {loss_ratio_num:.6f}")
            print(
                f"  Теоретически: {loss_ratio_th:.6f} (beta / (1 + beta) = 2/7 ≈ 0.285714)"
            )

    # Проверка законов сохранения
    print("Проверка законов сохранения (инвариантов модели):")
    print(
        f"  Макс. невязка закона изменения энергии |E_мех(t) + |A_тр(t)| - E(0)|: {res['energy_error']:.3e} Дж"
    )
    print(
        f"  Макс. невязка закона момента импульса  |L_O(t) - M_g*t - L_O(0)|:     {res['angular_momentum_error']:.3e} кг·м²/с"
    )
    print()


def run_all_experiments():
    figures_dir = Path(__file__).resolve().parent / "figures"
    figures_dir.mkdir(exist_ok=True)

    # Эксперимент 1: Горизонтальный стол, удар кием без начального вращения
    cfg1 = SCENARIOS["exp1_horizontal_forward_slip"]
    sim1 = RollingBodySimulator(
        cfg1["m"], cfg1["R"], cfg1["mu"], cfg1["alpha_deg"], cfg1["body_key"]
    )
    res1 = sim1.simulate(cfg1["x0"], cfg1["v0"], cfg1["omega0"], cfg1["t_max"])
    th1 = sim1.analytical_solution(res1["t"], cfg1["x0"], cfg1["v0"], cfg1["omega0"])
    print_experiment_summary(cfg1["title"], sim1, res1, th1)
    plot_single_experiment(
        res1,
        th1,
        cfg1["title"],
        cfg1["R"],
        filename=str(figures_dir / "exp1_horizontal_slip.png"),
    )

    # Эксперимент 2: Горизонтальный стол, удар с сильным обратным вращением
    cfg2 = SCENARIOS["exp2_horizontal_backspin"]
    sim2 = RollingBodySimulator(
        cfg2["m"], cfg2["R"], cfg2["mu"], cfg2["alpha_deg"], cfg2["body_key"]
    )
    res2 = sim2.simulate(cfg2["x0"], cfg2["v0"], cfg2["omega0"], cfg2["t_max"])
    th2 = sim2.analytical_solution(res2["t"], cfg2["x0"], cfg2["v0"], cfg2["omega0"])
    print_experiment_summary(cfg2["title"], sim2, res2, th2)
    plot_single_experiment(
        res2,
        th2,
        cfg2["title"],
        cfg2["R"],
        filename=str(figures_dir / "exp2_horizontal_backspin.png"),
    )

    # Эксперимент 3: Наклонная плоскость, бросок вверх со сменой режима движения
    cfg3 = SCENARIOS["exp3_incline_transition"]
    sim3 = RollingBodySimulator(
        cfg3["m"], cfg3["R"], cfg3["mu"], cfg3["alpha_deg"], cfg3["body_key"]
    )
    res3 = sim3.simulate(cfg3["x0"], cfg3["v0"], cfg3["omega0"], cfg3["t_max"])
    th3 = sim3.analytical_solution(res3["t"], cfg3["x0"], cfg3["v0"], cfg3["omega0"])
    print_experiment_summary(cfg3["title"], sim3, res3, th3)
    plot_single_experiment(
        res3,
        th3,
        cfg3["title"],
        cfg3["R"],
        filename=str(figures_dir / "exp3_incline_transition.png"),
    )

    # Эксперимент 4: Условия качения и сравнение ускорений цилиндров
    cfg4 = SCENARIOS["exp4_incline_comparison"]
    # 4А: Сплошной шар при малом трении (mu < mu_crit) и большом трении (mu > mu_crit)
    sim4_slip = RollingBodySimulator(
        cfg4["m"], cfg4["R"], cfg4["mu_low"], cfg4["alpha_deg"], "solid_sphere"
    )
    res4_slip = sim4_slip.simulate(0.0, 0.0, 0.0, cfg4["t_max"])
    res4_slip["mu"] = cfg4["mu_low"]
    res4_slip["R"] = cfg4["R"]

    sim4_roll = RollingBodySimulator(
        cfg4["m"], cfg4["R"], cfg4["mu_high"], cfg4["alpha_deg"], "solid_sphere"
    )
    res4_roll = sim4_roll.simulate(0.0, 0.0, 0.0, cfg4["t_max"])
    res4_roll["mu"] = cfg4["mu_high"]

    # 4Б: Сравнение полого цилиндра (beta=1) и сплошного цилиндра (beta=1/2)
    sim_hoop = RollingBodySimulator(
        cfg4["m"], cfg4["R"], cfg4["mu_high"], cfg4["alpha_deg"], "thin_hoop"
    )
    res_hoop = sim_hoop.simulate(0.0, 0.0, 0.0, cfg4["t_max"])

    sim_cyl = RollingBodySimulator(
        cfg4["m"], cfg4["R"], cfg4["mu_high"], cfg4["alpha_deg"], "solid_cylinder"
    )
    res_cyl = sim_cyl.simulate(0.0, 0.0, 0.0, cfg4["t_max"])

    a_hoop_num = res_hoop["a"][0]
    a_cyl_num = res_cyl["a"][0]
    ratio_num = a_hoop_num / a_cyl_num
    ratio_th = 3.0 / 4.0

    print("=" * 85)
    print(f" {cfg4['title']}")
    print("=" * 85)
    print(
        f"1) Сплошной шар на наклонной плоскости (alpha = {cfg4['alpha_deg']}°, mu_crit = {sim4_slip.mu_crit:.4f}):"
    )
    print(
        f"   - При mu = {cfg4['mu_low']} (< mu_crit): непрерывное скольжение, v_c(t_max) = {res4_slip['v_c'][-1]:.4f} м/с"
    )
    print(
        f"   - При mu = {cfg4['mu_high']} (> mu_crit): чистое качение без проскальзывания, v_c(t_max) = {res4_roll['v_c'][-1]:.2e} м/с"
    )
    print(
        "2) Сравнение ускорений полого цилиндра (beta=1) и сплошного цилиндра (beta=1/2):"
    )
    print(
        f"   - Ускорение полого цилиндра (численно):    a_пол = {a_hoop_num:.6f} м/с²"
    )
    print(f"   - Ускорение сплошного цилиндра (численно): a_спл = {a_cyl_num:.6f} м/с²")
    print(f"   - Отношение a_пол / a_спл (численно):       {ratio_num:.6f}")
    print(
        f"   - Отношение a_пол / a_спл (аналитически): {ratio_th:.6f} (3/4) | Погрешность: {abs(ratio_num - ratio_th):.2e}"
    )
    print("=" * 85)

    plot_incline_comparison(
        res4_slip,
        res4_roll,
        res_hoop,
        res_cyl,
        cfg4["title"],
        filename=str(figures_dir / "exp4_incline_comparison.png"),
    )


if __name__ == "__main__":
    run_all_experiments()
