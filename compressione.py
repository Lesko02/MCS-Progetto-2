import numpy as np
from scipy.fft import dct, idct
from PIL import Image

def dct2(A):
    """DCT2 ortonormale via scipy.fft (FFT-based)."""
    return dct(dct(A, type=2, norm='ortho', axis=0),
               type=2, norm='ortho', axis=1)


def idct2(A):
    """IDCT2 ortonormale via scipy.fft (FFT-based)."""
    return idct(idct(A, type=2, norm='ortho', axis=0),
                type=2, norm='ortho', axis=1)

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
    Hc = (H // F) * F 
    Wc = (W // F) * F
    img_cropped = img[:Hc, :Wc].astype(np.float64)

    mask = cutoff_mask(F, d).astype(np.float64)
    out = np.empty_like(img_cropped)

    for i in range(0, Hc, F):
        for j in range(0, Wc, F):
            block = img_cropped[i:i+F, j:j+F]
            c = dct2(block)
            c *= mask                   
            ff = idct2(c)
            out[i:i+F, j:j+F] = ff

    out = np.rint(out)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

def load_bmp_grayscale(path):
    """Carica un .bmp e lo restituisce come ndarray 2D uint8.

    Se l'immagine e' a colori viene convertita in scala di grigi.
    """
    im = Image.open(path)
    if im.mode != 'L':
        im = im.convert('L')
    return np.array(im, dtype=np.uint8)