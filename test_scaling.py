# Benchmark per verificare che le funzioni rispettino la specifica data

import numpy as np
from scipy.fft import dct

from dct_custom import dct_1D, dct_2D, idct_2D


# Test 1: riga 1x8
row = np.array([231, 32, 233, 161, 24, 71, 140, 245], dtype=float)
row_expected = np.array([4.01e+02, 6.60e+00, 1.09e+02, -1.12e+02,
                         6.54e+01, 1.21e+02, 1.16e+02, 2.88e+01])

c_row_nostro = dct_1D(row)
c_row_scipy = dct(row, type=2, norm='ortho')

print("CASO TEST 1: DCT 1D sulla prima riga")
print(f"Atteso     : {np.array2string(row_expected,  formatter={'float_kind': lambda x: f'{x: .2e}'})}")
print(f"Risultato nostro: {np.array2string(c_row_nostro,  formatter={'float_kind': lambda x: f'{x: .2e}'})}")
print(f"Risultato Scipy ('ortho'): {np.array2string(c_row_scipy,   formatter={'float_kind': lambda x: f'{x: .2e}'})}")
print(f"|custom - scipy| max = {np.max(np.abs(c_row_nostro - c_row_scipy)):.3e}")
print(f"|custom - atteso| max = {np.max(np.abs(c_row_nostro - row_expected)):.3e}")

#Test 2: blocco 8x8
block = np.array([
    [231,  32, 233, 161,  24,  71, 140, 245],
    [247,  40, 248, 245, 124, 204,  36, 107],
    [234, 202, 245, 167,   9, 217, 239, 173],
    [193, 190, 100, 167,  43, 180,   8,  70],
    [ 11,  24, 210, 177,  81, 243,   8, 112],
    [ 97, 195, 203,  47, 125, 114, 165, 181],
    [193,  70, 174, 167,  41,  30, 127, 245],
    [ 87, 149,  57, 192,  65, 129, 178, 228],
], dtype=float)

block_expected = np.array([
    [ 1.11e+03,  4.40e+01,  7.59e+01, -1.38e+02,  3.50e+00,  1.22e+02,  1.95e+02, -1.01e+02],
    [ 7.71e+01,  1.14e+02, -2.18e+01,  4.13e+01,  8.77e+00,  9.90e+01,  1.38e+02,  1.09e+01],
    [ 4.48e+01, -6.27e+01,  1.11e+02, -7.63e+01,  1.24e+02,  9.55e+01, -3.98e+01,  5.85e+01],
    [-6.99e+01, -4.02e+01, -2.34e+01, -7.67e+01,  2.66e+01, -3.68e+01,  6.61e+01,  1.25e+02],
    [-1.09e+02, -4.33e+01, -5.55e+01,  8.17e+00,  3.02e+01, -2.86e+01,  2.44e+00, -9.41e+01],
    [-5.38e+00,  5.66e+01,  1.73e+02, -3.54e+01,  3.23e+01,  3.34e+01, -5.81e+01,  1.90e+01],
    [ 7.88e+01, -6.45e+01,  1.18e+02, -1.50e+01, -1.37e+02, -3.06e+01, -1.05e+02,  3.98e+01],
    [ 1.97e+01, -7.81e+01,  9.72e-01, -7.23e+01, -2.15e+01,  8.13e+01,  6.37e+01,  5.90e+00],
])

c_nostro = dct_2D(block)
# DCT2 con scipy: prima sulle righe (axis=1) poi sulle colonne (axis=0), norm='ortho'
c_scipy = dct(dct(block, type=2, norm='ortho', axis=0),
              type=2, norm='ortho', axis=1)

print("\nCASO TEST 2: DCT2 sul blocco 8x8")
np.set_printoptions(precision=2, suppress=False, formatter={'float_kind': lambda x: f'{x: 9.2e}'})
print("\nRisultato nostro:")
print(c_nostro)
print("\nRisultato Scipy ('ortho'):")
print(c_scipy)
print(f"\n|custom - scipy|  max = {np.max(np.abs(c_nostro - c_scipy)):.3e}")
print(f"|custom - atteso| max = "
      f"{np.max(np.abs(c_nostro - block_expected)):.3e}")

# Verifica IDCT2 (round-trip)
back = idct_2D(c_nostro)
print(f"\n|idct_2D(dct_2D(block)) - block| max = {np.max(np.abs(back - block)):.3e}")