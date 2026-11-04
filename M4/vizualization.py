"""
visualization.py — Модуль визуализации результатов численных экспериментов (Задача М4).

========================================================================================
ПРИМЕЧАНИЕ О ГЕНЕРАЦИИ КОДА (согласно требованиям курса «Цифровизация физических процессов»):
Данный модуль отвечает исключительно за оформление графиков Matplotlib (визуализацию)
и был сгенерирован с использованием инструментов ИИ. Расчётная физическая часть и численное
решение дифференциальных уравнений находятся отдельно в модуле physics.py.
========================================================================================
"""

import matplotlib.pyplot as plt
import numpy as np


# Настройка общего стиля графиков
plt.rcParams["figure.figsize"] = (14, 9)
plt.rcParams["font.size"] = 10
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.35


def plot_single_experiment(sim_res, theory_res, title, R, filename=None):
    """
    Строит 4 информативных графика для одного сценария:
    1) Кинематика скоростей: v(t), omega(t)*R и скорость точки контакта v_c(t) + теория.
    2) Траектория x(t) и сила трения F_fr(t).
    3) Закон сохранения/изменения энергии: E_trans, E_rot, U_pot, E_mech, A_fr_loss и инвариант E_total.
    4) Закон сохранения/изменения момента импульса относительно плоскости L_O(t).
    """
    t = sim_res["t"]
    t_star = sim_res["t_transition"]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # 1. Скорости (численно vs аналитика)
    ax1 = axes[0, 0]
    ax1.plot(t, sim_res["v"], "b-", lw=2.2, label="Численно: $v(t)$ (скорость ЦМ)")
    ax1.plot(t, theory_res["v"], "k--", lw=1.5, label="Теория: $v_{теор}(t)$")
    ax1.plot(t, sim_res["omega"] * R, "g-", lw=2.2, label="Численно: $\\omega(t) \\cdot R$")
    ax1.plot(t, theory_res["omega"] * R, "r:", lw=1.8, label="Теория: $\\omega_{теор}(t) \\cdot R$")
    ax1.plot(t, sim_res["v_c"], "m-.", lw=1.5, label="Скорость контакта $v_c = v - \\omega R$")
    if t_star is not None:
        ax1.axvline(t_star, color="darkorange", ls="--", lw=1.8, label=f"Переход к качению $t^* = {t_star:.3f}$ с")
    ax1.axhline(0, color="gray", lw=0.8)
    ax1.set_title("Кинематика: поступательная и вращательная скорости")
    ax1.set_xlabel("Время $t$, с")
    ax1.set_ylabel("Скорость, м/с")
    ax1.legend(loc="best", fontsize=8.5)

    # 2. Координата x(t) и сила трения F_fr(t)
    ax2 = axes[0, 1]
    color_x = "tab:blue"
    ax2.plot(t, sim_res["x"], color=color_x, lw=2.2, label="Численно: $x(t)$")
    ax2.plot(t, theory_res["x"], "k--", lw=1.4, label="Теория: $x_{теор}(t)$")
    ax2.set_xlabel("Время $t$, с")
    ax2.set_ylabel("Координата центра масс $x$, м", color=color_x)
    ax2.tick_params(axis="y", labelcolor=color_x)

    ax2_sub = ax2.twinx()
    color_f = "tab:red"
    ax2_sub.plot(t, sim_res["F_fr"], color=color_f, lw=1.8, ls="-", alpha=0.85, label="Сила трения $F_{тр}(t)$")
    ax2_sub.set_ylabel("Сила трения $F_{тр}$, Н", color=color_f)
    ax2_sub.tick_params(axis="y", labelcolor=color_f)
    ax2_sub.grid(False)

    if t_star is not None:
        ax2.axvline(t_star, color="darkorange", ls="--", lw=1.8)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_sub.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="best", fontsize=8.5)
    ax2.set_title("Перемещение $x(t)$ и скачок силы трения при смене режима")

    # 3. Закон сохранения (изменения) энергии
    ax3 = axes[1, 0]
    ax3.plot(t, sim_res["E_trans"], label="$E_{пост} = mv^2/2$", lw=1.6)
    ax3.plot(t, sim_res["E_rot"], label="$E_{вр} = I\\omega^2/2$", lw=1.6)
    if np.max(np.abs(sim_res["U_pot"])) > 1e-6:
        ax3.plot(t, sim_res["U_pot"], label="$U_{пот} = -mgx\\sin\\alpha$", lw=1.6)
    ax3.plot(t, sim_res["E_mech"], "b-", lw=2.2, label="Полная мех. энергия $E_{мех}(t)$")
    ax3.plot(t, sim_res["A_fr_loss"], "r-.", lw=1.8, label="Потери на трение $|A_{тр}(t)|$")
    ax3.plot(
        t,
        sim_res["E_total_conserved"],
        "k--",
        lw=2.2,
        label=f"Инвариант $E_{{мех}} + |A_{{тр}}| = const$ (ошибка: {sim_res['energy_error']:.1e} Дж)",
    )
    if t_star is not None:
        ax3.axvline(t_star, color="darkorange", ls="--", lw=1.8)
    ax3.set_title("Баланс энергии и проверка закона изменения энергии")
    ax3.set_xlabel("Время $t$, с")
    ax3.set_ylabel("Энергия, Дж")
    ax3.legend(loc="best", fontsize=8.5)

    # 4. Закон сохранения (изменения) момента импульса относительно плоскости
    ax4 = axes[1, 1]
    ax4.plot(
        t,
        sim_res["L_O"],
        "purple",
        lw=2.2,
        label="Момент импульса $L_O(t) = mvR + I\\omega$",
    )
    ax4.plot(
        t,
        sim_res["L_O_invariant"],
        "k--",
        lw=2.0,
        label=f"Инвариант $L_O(t) - M_g t = const$ (ошибка: {sim_res['angular_momentum_error']:.1e})",
    )
    if t_star is not None:
        ax4.axvline(t_star, color="darkorange", ls="--", lw=1.8)
    ax4.set_title("Проверка закона момента импульса относительно точки на плоскости")
    ax4.set_xlabel("Время $t$, с")
    ax4.set_ylabel("Момент импульса $L_O$, кг·м²/с")
    ax4.legend(loc="best", fontsize=8.5)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    if filename:
        plt.savefig(filename, dpi=150)
    plt.show()


def plot_incline_comparison_and_t52(res_slip, res_sphere_roll, res_hollow, res_cyl, title, filename=None):
    """
    Визуализирует результаты Эксперимента 4:
    - Слева: влияние коэффициента трения на наклонной плоскости (mu < mu_crit vs mu > mu_crit).
    - Справа: проверка задачи Т5.2 (скатывание без проскальзывания полого мяча и сплошного цилиндра).
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    # График 1: Сравнение скольжения (mu < mu_crit) и чистого качения (mu > mu_crit)
    ax1 = axes[0]
    t = res_slip["t"]
    R = res_slip["R"]
    ax1.plot(t, res_slip["v"], "r-", lw=2.2, label=f"Скольжение ($\\mu={res_slip['mu']} < \\mu_{{кр}}$): $v(t)$")
    ax1.plot(t, res_slip["omega"] * R, "r--", lw=1.8, label=f"Скольжение ($\\mu={res_slip['mu']}$): $\\omega(t)R$")
    ax1.plot(t, res_sphere_roll["v"], "b-", lw=2.2, label=f"Без проскальзывания ($\\mu={res_sphere_roll['mu']} > \\mu_{{кр}}$): $v(t) = \\omega(t)R$")
    ax1.set_title("Режимы скатывания шара в зависимости от $\\mu$ и $\\mu_{кр}$")
    ax1.set_xlabel("Время $t$, с")
    ax1.set_ylabel("Скорость, м/с")
    ax1.legend(loc="best")

    # График 2: Проверка задачи Т5.2 (Полый мяч vs Сплошной цилиндр)
    ax2 = axes[1]
    ax2.plot(res_cyl["t"], res_cyl["v"], "g-", lw=2.2, label=f"Сплошной цилиндр ($\\beta=1/2$): $a_{{спл}} = {res_cyl['a'][0]:.4f}$ м/с²")
    ax2.plot(res_hollow["t"], res_hollow["v"], "m-", lw=2.2, label=f"Полый мяч ($\\beta=2/3$): $a_{{пол}} = {res_hollow['a'][0]:.4f}$ м/с²")
    ax2.plot(res_cyl["t"], res_cyl["x"], "g--", lw=1.8, label="Путь цилиндра $x_{спл}(t)$, м")
    ax2.plot(res_hollow["t"], res_hollow["x"], "m--", lw=1.8, label="Путь полого мяча $x_{пол}(t)$, м")

    ratio_num = res_hollow["a"][0] / res_cyl["a"][0]
    ax2.set_title(f"Задача Т5.2: $a_{{пол}} / a_{{спл}} = {ratio_num:.6f}$ (теория: $3/4 = 0.750000$)")
    ax2.set_xlabel("Время $t$, с")
    ax2.set_ylabel("Скорость (м/с) / Координата (м)")
    ax2.legend(loc="best")

    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    if filename:
        plt.savefig(filename, dpi=150)
    plt.show()