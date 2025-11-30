from typing import Callable, Sequence

from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt


def plot_fit(my_func: Callable[..., np.ndarray], x: np.ndarray, y: np.ndarray, *,
             x_err: float | list[float] | np.ndarray | None = None,
             y_err: float | list[float] | np.ndarray | None = None, init_guess: Sequence[float] | None = None,
             font_size: int = 14, x_label: str = "Independent Variable (units)",
             y_label: str = "Dependent Variable (units)", title: str = "Graph") -> tuple[list[float], list[float]]:
    plt.rcParams.update({"font.size": font_size})
    plt.rcParams["figure.figsize"] = 10, 9
    popt, pcov = curve_fit(my_func, x, y, p0=init_guess)
    puncert = np.sqrt(np.diagonal(pcov))
    for i in range(len(popt)):
        print(popt[i], "+/-", puncert[i])
    start = min(x)
    stop = max(x)
    xs = np.arange(start, stop, (stop - start) / 1000)
    curve = my_func(xs, *popt)
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [2, 1]})
    ax1.errorbar(x, y, yerr=y_err, xerr=x_err, fmt=".", label="data", color="black", ecolor="#cb6627", zorder=0)
    ax1.plot(xs, curve, label="best fit", color="#5895a0", zorder=1)
    ax1.legend(loc="upper center")
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label)
    residuals = y - my_func(x, *popt)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    print(r_squared)
    ax2.errorbar(x, residuals, yerr=y_err, xerr=x_err, fmt=".", color="black", ecolor="#cb6627")
    ax2.axhline(y=0, color="black")
    ax2.set_xlabel(x_label)
    ax2.set_ylabel("Residuals")
    fig.tight_layout()
    plt.show()
    fig.savefig(f"{title.lower()}.png")
