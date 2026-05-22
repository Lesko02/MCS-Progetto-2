"""
PARTE 1 - Confronto dei tempi di esecuzione DCT2 vs DCT2
della libreria (scipy.fft.dct, basata su FFT).

Si attendono tempi proporzionali a:
    - N^3        per la DCT2 (algoritmo matriciale)
    - N^2 log N  per la versione fast (FFT)

Output:
    - stampa di una tabella con i tempi
    - grafico in scala semilogaritmica (solo ordinate)
    - salvataggio dei dati in 'times.csv'
    - salvataggio del grafico in 'times.png'
"""

import time
import csv
import pathlib

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dct
from pathlib import Path
from dct_custom import dct_2D as dct2_custom

# DCT2 della libreria: scipy.fft.dct e' la versione FFT-based.
# Per la 2D applichiamo la dct prima per colonne (axis=0) e poi per righe
# (axis=1) usando norm='ortho'.
def dct2_scipy(A):
    return dct(dct(A, type=2, norm='ortho', axis=0),
               type=2, norm='ortho', axis=1)


def time_call(func, A, repeats=1):
    """Esegue func(A) e restituisce il tempo medio in secondi su `repeats`
    chiamate. Tutte le dimensioni considerate vengono testate sulla stessa
    matrice."""

    func(A)
    best = np.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        func(A)
        t1 = time.perf_counter()
        best = min(best, t1 - t0)
    return best


def main():
    # Dimensioni N crescenti.
    sizes = [8, 16, 32, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512, 768, 1024, 1536, 2048]

    rng = np.random.default_rng(0)

    # repeats: poche per il custom (e' lento), piu' per scipy (sub-ms)
    repeats_custom = 5
    repeats_scipy  = 10

    print(f"{'N':>6} | {'custom [s]':>12} | {'scipy [s]':>12} | "
          f"{'speedup':>9}")
    print("-" * 52)

    times_custom = {}
    times_scipy  = {}

    for N in sizes:
        A = rng.standard_normal((N, N))
        t_custom = time_call(dct2_custom, A, repeats=repeats_custom)
        t_scipy  = time_call(dct2_scipy, A, repeats=repeats_scipy)
        times_custom[N] = t_custom
        times_scipy[N]  = t_scipy
        print(f"{N:>6} | {t_custom:>12.4e} | {t_scipy:>12.4e} | "
              f"{t_custom / t_scipy:>9.1f}x")

    percorso_csv = Path('Results/times.csv')

    # Crea la cartella 'Results' (e le eventuali cartelle padre) se non esiste
    percorso_csv.parent.mkdir(parents=True, exist_ok=True)

    # Salvataggio CSV
    with open('Results/times.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['N', 'time_custom_s', 'time_scipy_s'])
        all_N = sorted(set(times_custom.keys()) | set(times_scipy.keys()))
        for N in all_N:
            w.writerow([N,
                        times_custom.get(N, ''),
                        times_scipy.get(N, '')])

    # Grafico semilog-y (solo ordinate in log)
    Ns_c = sorted(times_custom.keys())
    Ns_s = sorted(times_scipy.keys())
    Tc = np.array([times_custom[N] for N in Ns_c])
    Ts = np.array([times_scipy[N]  for N in Ns_s])

    # rette di riferimento N^3 e N^2 log(N) calibrate su un punto centrale
    Ns_c_arr = np.array(Ns_c, dtype=float)
    Ns_s_arr = np.array(Ns_s, dtype=float)
    ic = len(Ns_c) // 2          # punto di calibrazione per la curva custom
    is_ = len(Ns_s) // 2         # punto di calibrazione per scipy
    ref_N3   = Tc[ic] * (Ns_c_arr / Ns_c[ic]) ** 3
    ref_N2lg = (Ts[is_] *
                (Ns_s_arr ** 2 * np.log(Ns_s_arr)) /
                (Ns_s[is_] ** 2 * np.log(Ns_s[is_])))

    plt.figure(figsize=(9, 6))
    plt.semilogy(Ns_c, Tc,  'o-', label='DCT2 propria (matriciale, O(N^3))',
                 color='C3', markersize=6, linewidth=1.6)
    plt.semilogy(Ns_s, Ts,  's-', label='DCT2 libreria scipy.fft (FFT, O(N^2 log N))',
                 color='C0', markersize=6, linewidth=1.6)
    plt.semilogy(Ns_c, ref_N3,  '--', color='C3', alpha=0.55, label='riferimento $\\sim N^3$')
    plt.semilogy(Ns_s, ref_N2lg,'--', color='C0', alpha=0.55, label='riferimento $\\sim N^2 \\log N$')

    plt.xlabel('N  (lato della matrice quadrata)')
    plt.ylabel('tempo di esecuzione [s]  (scala logaritmica)')
    plt.title('Tempi di esecuzione DCT2: implementazione propria vs scipy.fft')
    plt.grid(True, which='both', linestyle=':', alpha=0.55)
    plt.legend(loc='upper left')
    plt.tight_layout()
    # Creazione sicura della cartella e salvataggio
    percorso_plot = Path('Plots/times.png')
    percorso_plot.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(percorso_plot, dpi=140)

    print(f"\nGrafico salvato in '{percorso_plot}'")
    print("Dati salvati  in 'times.csv'")


if __name__ == '__main__':
    main()