# Esperimenti riproducibili, esegue la compressione con varie soglie sull'intera
# cartella di Immagini

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from compressione import compress_image, load_bmp_grayscale


# Metriche
def psnr_uint8(orig, comp):
    """Calcola il PSNR in dB e l'MSE tra due immagini."""
    diff = orig.astype(np.float64) - comp.astype(np.float64)
    mse = float(np.mean(diff * diff))
    if mse == 0:
        return float('inf'), 0.0
    return 10.0 * np.log10(255.0 ** 2 / mse), mse


def coeffs_kept(F, d):
    """Calcola il numero di coefficienti con k+l < d in un blocco F x F."""
    k = np.arange(F).reshape(F, 1)
    l = np.arange(F).reshape(1, F)
    return int(np.sum((k + l) < d))


def run_experiment(name, img, configs, plot_dir='Plots', csv_dir='Results'):
    """Esegue la compressione per tutte le configurazioni e salva output e metriche."""
    os.makedirs(plot_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    H, W = img.shape

    rows = []
    n = len(configs)
    
    # Dimensionamento dinamico del grafico in base al numero di configurazioni
    fig, axes = plt.subplots(n, 2, figsize=(8, 3.5 * n))
    if n == 1:
        axes = axes.reshape(1, -1)

    for i, (F, d) in enumerate(configs):
        # Compressione
        out = compress_image(img, F, d)
        Hc, Wc = out.shape
        
        # Metriche
        psnr, mse = psnr_uint8(img[:Hc, :Wc], out)
        kept = coeffs_kept(F, d)
        kept_frac = kept / (F * F)
        rows.append([F, d, kept, F * F, kept_frac, mse, psnr])

        # Plotting (colonna sinistra: Originale tagliata, colonna destra: Compressa)
        axes[i, 0].imshow(img[:Hc, :Wc], cmap='gray', vmin=0, vmax=255)
        axes[i, 0].set_title(f"Originale ({Wc}×{Hc})")
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(out, cmap='gray', vmin=0, vmax=255)
        axes[i, 1].set_title(
            f"F={F}, d={d} ({100*kept_frac:.1f}% coeff.)\n"
            f"PSNR: {psnr:.1f} dB  |  MSE: {mse:.1f}"
        )
        axes[i, 1].axis('off')

    # Salvataggio immagine
    plt.suptitle(f"Analisi Compressione: {name}", fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    fig_path = os.path.join(plot_dir, f"plot_{name}.png")
    plt.savefig(fig_path, dpi=130, bbox_inches='tight')
    plt.close(fig)

    # Salvataggio CSV
    csv_path = os.path.join(csv_dir, f"dati_{name}.csv")
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['F', 'd', 'coeff_tenuti', 'coeff_totali', 'frazione_tenuti', 'MSE', 'PSNR_dB'])
        w.writerows(rows)

    # Output in console
    print(f"Salvato plot: {fig_path}")
    print(f"Salvato CSV:  {csv_path}")
    for F, d, kept, tot, frac, mse, psnr in rows:
        print(f"    F={F:>3} d={d:>3} | coeff={kept:>4}/{tot:<4} "
              f"({100*frac:>5.1f}%) | MSE={mse:>7.2f} | PSNR={psnr:>5.2f} dB")


def main():
    img_dir = "Immagini"
    plot_dir = "Plots"
    csv_dir = "Results"
    
    # Check esistenza cartella
    if not os.path.exists(img_dir):
        print(f"ERRORE: Impossibile trovare la cartella '{img_dir}'.")
        print("Assicurati che sia nella stessa directory di questo script.")
        return

    # Definizione delle coppie (F, d) da testare
    # Potete modificare queste tuple a piacimento per la relazione
    configs = [
        (8, 14),   # Compressione quasi nulla
        (8, 7),    # Compressione media
        (8, 3),    # Compressione aggressiva
        (16, 15),  # Blocco più grande, compressione media
        (32, 10)   # Blocco enorme, compressione elevata
    ]

    # Cerca tutti i file .bmp nella cartella
    bmp_files = [f for f in os.listdir(img_dir) if f.lower().endswith('.bmp')]
    
    if not bmp_files:
        print(f"Nessuna immagine .bmp trovata nella cartella '{img_dir}'.")
        return

    print(f"Trovate {len(bmp_files)} immagini. Inizio compressione:\n" + "-"*50)

    for filename in bmp_files:
        img_path = os.path.join(img_dir, filename)
        name = os.path.splitext(filename)[0]
        
        print(f"\nImmagine: {filename}")
        try:
            img = load_bmp_grayscale(img_path)
            run_experiment(name, img, configs, plot_dir=plot_dir, csv_dir=csv_dir)
        except Exception as e:
            print(f" ERRORE nell'elaborazione di {filename}: {e}")

if __name__ == '__main__':
    main()