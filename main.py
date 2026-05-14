import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path

l = 3.5
Ns = [3, 10, 30]
out_dir = Path("fourier_plots")
out_dir.mkdir(exist_ok=True)

x = np.linspace(0, l, 4000)
def f_original(x):
    return np.where(x < l, np.floor(x), 3.0)

def full_a0():
    return 18 / 7

def full_an(n):
    return -(np.sin(4 * np.pi * n / 7) +
             np.sin(8 * np.pi * n / 7) +
             np.sin(12 * np.pi * n / 7)) / (np.pi * n)

def full_bn(n):
    return (np.cos(4 * np.pi * n / 7) +
            np.cos(8 * np.pi * n / 7) +
            np.cos(12 * np.pi * n / 7) - 3) / (np.pi * n)

def cos_a0():
    return 18 / 7

def cos_an(n):
    return -(2 / (np.pi * n)) * (
        np.sin(2 * np.pi * n / 7) +
        np.sin(4 * np.pi * n / 7) +
        np.sin(6 * np.pi * n / 7)
    )

def sin_bn(n):
    return (2 / (np.pi * n)) * (
        np.cos(2 * np.pi * n / 7) +
        np.cos(4 * np.pi * n / 7) +
        np.cos(6 * np.pi * n / 7) -
        3 * (-1) ** n
    )

def S_full(x, N):
    s = np.full_like(x, full_a0() / 2, dtype=float)
    for n in range(1, N + 1):
        s += full_an(n) * np.cos(4 * np.pi * n * x / 7)
        s += full_bn(n) * np.sin(4 * np.pi * n * x / 7)
    return s

def S_cos(x, N):
    s = np.full_like(x, cos_a0() / 2, dtype=float)
    for n in range(1, N + 1):
        s += cos_an(n) * np.cos(2 * np.pi * n * x / 7)
    return s

def S_sin(x, N):
    s = np.zeros_like(x, dtype=float)
    for n in range(1, N + 1):
        s += sin_bn(n) * np.sin(2 * np.pi * n * x / 7)
    return s

def make_plot(x, y_true, y_sum, title, filename):
    plt.figure(figsize=(10, 6))
    plt.step(x, y_true, where="post", label="f(x) = [x]", linewidth=2)
    plt.plot(x, y_sum, label="Частичная сумма", linewidth=2)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(0, l)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

def main():
    total_start = time.perf_counter()
    y_true = f_original(x)

    timings = []
    for N in Ns:
        start = time.perf_counter()
        y = S_full(x, N)
        filename = out_dir / f"full_N{N}.png"
        make_plot(
            x, y_true, y,
            f"Полный ряд Фурье, N = {N}",
            filename
        )
        elapsed = time.perf_counter() - start
        timings.append(("Полный ряд", N, elapsed))
    for N in Ns:
        start = time.perf_counter()
        y = S_cos(x, N)
        filename = out_dir / f"cosine_N{N}.png"
        make_plot(
            x, y_true, y,
            f"Ряд Фурье по косинусам, N = {N}",
            filename
        )
        elapsed = time.perf_counter() - start
        timings.append(("Косинусный ряд", N, elapsed))

    for N in Ns:
        start = time.perf_counter()
        y = S_sin(x, N)
        filename = out_dir / f"sine_N{N}.png"
        make_plot(
            x, y_true, y,
            f"Ряд Фурье по синусам, N = {N}",
            filename
        )
        elapsed = time.perf_counter() - start
        timings.append(("Синусный ряд", N, elapsed))

    total_time = time.perf_counter() - total_start

    print("Графики сохранены в папку:", out_dir.resolve())
    print()
    print("Время построения графиков:")
    for name, N, elapsed in timings:
        print(f"{name:16s} | N = {N:2d} | {elapsed:.6f} сек")
    print()
    print(f"Общее время работы программы: {total_time:.6f} сек")

    with open("fourier_results.txt", "w", encoding="utf-8") as f:
        f.write("Численный метод. Ряды Фурье для f(x) = [x] на [0, 3.5]\n\n")
        f.write("Построены графики частичных сумм для N = 3, 10, 30.\n\n")
        f.write("Время построения графиков:\n")
        for name, N, elapsed in timings:
            f.write(f"{name:16s} | N = {N:2d} | {elapsed:.6f} сек\n")
        f.write(f"\nОбщее время работы программы: {total_time:.6f} сек\n")

if __name__ == "__main__":
    main()