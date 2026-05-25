import numpy as np

def compute_D(N):
    """
    Costruisce la matrice D (N x N) della DCT-II ortonormale.

    D[k, i] = alpha_k * cos( k * pi * (2 i + 1) / (2 N) ),     k, i = 0..N-1
    con  alpha_0 = 1/sqrt(N),  alpha_k = sqrt(2/N) per k >= 1.

    NOTA: rispetto al codice MATLAB del prof gli indici sono 0-based,
    quindi al posto di (k-1) e (2i-1) compaiono k e (2i+1).
    """
    alpha = np.empty(N)
    alpha[0] = 1.0 / np.sqrt(N)
    alpha[1:] = np.sqrt(2.0 / N)

    k = np.arange(N).reshape(N, 1)   # indice di riga
    i = np.arange(N).reshape(1, N)   # indice di colonna
    D = alpha.reshape(N, 1) * np.cos(k * np.pi * (2 * i + 1) / (2 * N))
    return D

def dct_1D(f_vect):
    """DCT-II ortonormale di un vettore (fatta in casa: c = D * f)."""
    f_vect = np.asarray(f_vect, dtype=float).reshape(-1)
    N = f_vect.size
    D = compute_D(N)
    return D @ f_vect

def idct_1D(c_vect):
    """IDCT-II ortonormale di un vettore (f = D^T * c)."""
    c_vect = np.asarray(c_vect, dtype=float).reshape(-1)
    N = c_vect.size
    D = compute_D(N)
    return D.T @ c_vect

def dct_2D(f_mat):
    """
    DCT2: prima per colonne, poi per righe (come nelle
    note, Osservazione 7.2 e Figura 17).
    Costo: O(N^3).
    """
    f_mat = np.asarray(f_mat, dtype=float)
    N = f_mat.shape[0]
    D = compute_D(N)

    c_mat = f_mat.copy()
    # DCT su ogni colonna
    for j in range(N):
        c_mat[:, j] = D @ c_mat[:, j]
    # DCT su ogni riga
    for i in range(N):
        c_mat[i, :] = D @ c_mat[i, :]
    return c_mat

def idct_2D(c_mat):
    """IDCT2 'fatta in casa': D^T sulle colonne e poi sulle righe."""
    c_mat = np.asarray(c_mat, dtype=float)
    N = c_mat.shape[0]
    D = compute_D(N)
    Dt = D.T

    f_mat = c_mat.copy()
    for j in range(N):
        f_mat[:, j] = Dt @ f_mat[:, j]
    for i in range(N):
        f_mat[i, :] = Dt @ f_mat[i, :]
    return f_mat