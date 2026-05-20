"""
PARTE 1 - Confronto dei tempi di esecuzione DCT2 'fatta in casa' vs DCT2
della libreria (scipy.fft.dct, basata su FFT).

Si attendono tempi proporzionali a:
    - N^3        per la DCT2 'fatta in casa' (algoritmo matriciale)
    - N^2 log N  per la versione fast (FFT)

Output:
    - stampa di una tabella con i tempi
    - grafico in scala semilogaritmica (solo ordinate)
    - salvataggio dei dati in 'parte1_times.csv'
    - salvataggio del grafico in 'parte1_times.png'
"""

import time
import csv

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import dct

from dct_custom import dct_2D as dct2_custom


# ---------------------------------------------------------------------------
# DCT2 della libreria: scipy.fft.dct e' la versione FFT-based.
# Per la 2D applichiamo la dct prima per colonne (axis=0) e poi per righe
# (axis=1) usando norm='ortho' (stessa convenzione di scaling del prof).
# ---------------------------------------------------------------------------
def dct2_scipy(A):
    return dct(dct(A, type=2, norm='ortho', axis=0),
               type=2, norm='ortho', axis=1)


def time_call(func, A, repeats=1):
    """Esegue func(A) e restituisce il tempo medio in secondi su `repeats`
    chiamate. Tutte le dimensioni considerate vengono testate sulla stessa
    matrice."""
    # warm-up (la prima chiamata di scipy puo' includere overhead di import)
    func(A)
    best = np.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        func(A)
        t1 = time.perf_counter()
        best = min(best, t1 - t0)
    return best


def main():
    # Dimensioni N crescenti. Per la DCT2 fatta in casa il costo e' O(N^3),
    # quindi non si va oltre qualche centinaio se vogliamo tempi ragionevoli.
    sizes_custom = [8, 16, 32, 64, 96, 128, 160, 192, 224, 256, 320, 384, 448, 512] + [768, 1024, 1536, 2048]
    # scipy e' ordini di grandezza piu' veloce: arriviamo piu' in alto.
    sizes_scipy  = sizes_custom + [768, 1024, 1536, 2048]

    rng = np.random.default_rng(0)

    # repeats: poche per il custom (e' lento), piu' per scipy (sub-ms)
    repeats_custom = 5
    repeats_scipy  = 10

    print(f"{'N':>6} | {'custom [s]':>12} | {'scipy [s]':>12} | "
          f"{'speedup':>9}")
    print("-" * 52)

    times_custom = {}
    times_scipy  = {}

    # Prima eseguiamo il custom solo per i sizes_custom...
    for N in sizes_custom:
        A = rng.standard_normal((N, N))
        t_custom = time_call(dct2_custom, A, repeats=repeats_custom)
        t_scipy  = time_call(dct2_scipy,  A, repeats=repeats_scipy)
        times_custom[N] = t_custom
        times_scipy[N]  = t_scipy
        print(f"{N:>6} | {t_custom:>12.4e} | {t_scipy:>12.4e} | "
              f"{t_custom / t_scipy:>9.1f}x")

    # ...e poi proseguiamo con scipy soltanto sui sizes piu' grandi
    for N in sizes_scipy:
        if N in times_scipy:
            continue
        A = rng.standard_normal((N, N))
        t_scipy = time_call(dct2_scipy, A, repeats=repeats_scipy)
        times_scipy[N] = t_scipy
        print(f"{N:>6} | {'-- skip --':>12} | {t_scipy:>12.4e} | {'--':>9}")

    # ----- Salvataggio CSV -----
    with open('Results/parte1_times.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['N', 'time_custom_s', 'time_scipy_s'])
        all_N = sorted(set(times_custom.keys()) | set(times_scipy.keys()))
        for N in all_N:
            w.writerow([N,
                        times_custom.get(N, ''),
                        times_scipy.get(N, '')])

    # ----- Grafico semilog-y (solo ordinate in log) -----
    Ns_c = sorted(times_custom.keys())
    Ns_s = sorted(times_scipy.keys())
    Tc = np.array([times_custom[N] for N in Ns_c])
    Ts = np.array([times_scipy[N]  for N in Ns_s])

    # rette di riferimento N^3 e N^2 log(N) calibrate su un punto centrale
    # (per esibire l'andamento asintotico evitando l'overhead per N piccoli)
    Ns_c_arr = np.array(Ns_c, dtype=float)
    Ns_s_arr = np.array(Ns_s, dtype=float)
    ic = len(Ns_c) // 2          # punto di calibrazione per la curva custom
    is_ = len(Ns_s) // 2         # punto di calibrazione per scipy
    ref_N3   = Tc[ic] * (Ns_c_arr / Ns_c[ic]) ** 3
    ref_N2lg = (Ts[is_] *
                (Ns_s_arr ** 2 * np.log(Ns_s_arr)) /
                (Ns_s[is_] ** 2 * np.log(Ns_s[is_])))

    plt.figure(figsize=(9, 6))
    plt.semilogy(Ns_c, Tc,  'o-', label='DCT2 fatta in casa (matriciale, O(N^3))',
                 color='C3', markersize=6, linewidth=1.6)
    plt.semilogy(Ns_s, Ts,  's-', label='DCT2 libreria scipy.fft (FFT, O(N^2 log N))',
                 color='C0', markersize=6, linewidth=1.6)
    plt.semilogy(Ns_c, ref_N3,  '--', color='C3', alpha=0.55, label='riferimento $\\sim N^3$')
    plt.semilogy(Ns_s, ref_N2lg,'--', color='C0', alpha=0.55, label='riferimento $\\sim N^2 \\log N$')

    plt.xlabel('N  (lato della matrice quadrata)')
    plt.ylabel('tempo di esecuzione [s]  (scala logaritmica)')
    plt.title('Parte 1 — Tempi di esecuzione DCT2: implementazione propria vs scipy.fft')
    plt.grid(True, which='both', linestyle=':', alpha=0.55)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig('Plots/parte1_times.png', dpi=140)
    print("\nGrafico salvato in 'parte1_times.png'")
    print("Dati salvati  in 'parte1_times.csv'")


if __name__ == '__main__':
    main()