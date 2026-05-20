"""
PARTE 2 - logica di compressione tipo JPEG (senza matrice di quantizzazione).

Procedimento (per immagini in toni di grigio):
    1. l'utente sceglie F (lato del macro-blocco) e d (soglia di taglio,
       0 <= d <= 2F-2);
    2. l'immagine viene divisa in blocchi F x F, scartando le righe/colonne
       eccedenti in basso e a destra;
    3. ad ogni blocco f si applica la DCT2 (versione fast, scipy.fft.dct);
    4. nel blocco c = DCT2(f) si pongono a zero tutti i coefficienti c[k, l]
       con k + l >= d (gli indici partono da 0; se d = 0 si annulla tutto,
       se d = 2F-2 si annulla solo c[F-1, F-1]);
    5. si calcola ff = IDCT2(c) modificato;
    6. si arrotonda ff all'intero piu' vicino e si "satura" in [0, 255];
    7. si rincollano i blocchi nell'ordine giusto per ricomporre l'immagine.
"""

import numpy as np
from scipy.fft import dct, idct
from PIL import Image


# ---------------------------------------------------------------------------
# DCT2 / IDCT2 della libreria (FFT-based) -- versione "fast"
# ---------------------------------------------------------------------------
def dct2(A):
    """DCT2 ortonormale via scipy.fft (FFT-based)."""
    return dct(dct(A, type=2, norm='ortho', axis=0),
               type=2, norm='ortho', axis=1)


def idct2(A):
    """IDCT2 ortonormale via scipy.fft (FFT-based)."""
    return idct(idct(A, type=2, norm='ortho', axis=0),
                type=2, norm='ortho', axis=1)


# ---------------------------------------------------------------------------
# Maschera di taglio: True dove il coefficiente deve essere conservato
# ---------------------------------------------------------------------------
def cutoff_mask(F, d):
    """
    Maschera FxF booleana: True dove (k + l) < d, False dove (k + l) >= d.

    Significato: i coefficienti c[k, l] con k + l >= d vengono azzerati.
    Per d = 0 si azzera tutto, per d = 2F-2 si azzera solo c[F-1, F-1].
    """
    k = np.arange(F).reshape(F, 1)
    l = np.arange(F).reshape(1, F)
    m = (k + l) < d
    return m


# ---------------------------------------------------------------------------
# Compressione blocco-per-blocco
# ---------------------------------------------------------------------------
def compress_image(img, F, d):
    """
    img : ndarray 2D (uint8 o float) contenente l'immagine in scala di grigi.
    F   : intero positivo, lato del macro-blocco.
    d   : intero in [0, 2F - 2], soglia di taglio.

    Restituisce l'immagine ricomposta come ndarray uint8 (stesse dimensioni
    della porzione divisibile in blocchi F x F; gli "avanzi" sono scartati).
    """
    if F <= 0:
        raise ValueError("F deve essere un intero positivo.")
    if not (0 <= d <= 2 * F - 2):
        raise ValueError(f"d deve essere in [0, 2F-2] = [0, {2*F-2}].")

    H, W = img.shape
    Hc = (H // F) * F   # dimensioni "tagliate" (scartando gli avanzi)
    Wc = (W // F) * F
    img_cropped = img[:Hc, :Wc].astype(np.float64)

    mask = cutoff_mask(F, d).astype(np.float64)  # 0/1 in float
    out = np.empty_like(img_cropped)

    for i in range(0, Hc, F):
        for j in range(0, Wc, F):
            block = img_cropped[i:i+F, j:j+F]
            c = dct2(block)
            c *= mask                          # azzera frequenze k+l >= d
            ff = idct2(c)
            out[i:i+F, j:j+F] = ff

    # arrotondamento e clipping a [0, 255]
    out = np.rint(out)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# Caricamento .bmp in toni di grigio
# ---------------------------------------------------------------------------
def load_bmp_grayscale(path):
    """Carica un .bmp e lo restituisce come ndarray 2D uint8.

    Se l'immagine e' a colori viene convertita in scala di grigi.
    """
    im = Image.open(path)
    if im.mode != 'L':
        im = im.convert('L')
    return np.array(im, dtype=np.uint8)