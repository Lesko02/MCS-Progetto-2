# Progetto 2 — Compressione di immagini tramite DCT

Metodi del Calcolo Scientifico, A.A. 2025/2026.

## Requisiti

- Python 3.10+ (testato con 3.12)
- Pacchetti: `numpy`, `scipy`, `matplotlib`, `Pillow`
  - tkinter è incluso nella libreria standard di Python su Windows/macOS;
    su Linux può servire installare il pacchetto di sistema
    `python3-tk` (es. `sudo apt install python3-tk` su Ubuntu/Debian).

Installazione veloce delle dipendenze:

    pip install numpy scipy matplotlib Pillow

## File del progetto

| File                          | Cosa fa                                                                |
|-------------------------------|------------------------------------------------------------------------|
| `dct_custom.py`               | DCT/IDCT 1D e 2D "fatte in casa" (Parte 1)                             |
| `test_scaling.py`             | Verifica i due casi test della traccia (riga 1×8 e blocco 8×8)         |
| `parte1_confronto_tempi.py`   | Confronto tempi DCT2 custom vs scipy.fft + grafico semilog             |
| `parte2_compressione.py`      | Motore di compressione (DCT2/IDCT2 fast, cutoff_mask, compress_image)  |
| `parte2_gui.py`               | Interfaccia grafica tkinter (Parte 2)                                  |
| `parte2_esperimenti.py`       | Esperimenti riproducibili su immagini sintetiche o file forniti        |
| `Relazione_Progetto2_DCT.docx`| Relazione finale                                                       |

## Come si usa

1. **Verifica lo scaling sui casi test del docente**

       python test_scaling.py

   Deve mostrare un errore custom-vs-scipy dell'ordine di 1e-13.

2. **Parte 1 — confronto dei tempi**

       python parte1_confronto_tempi.py

   Stampa la tabella dei tempi e salva `parte1_times.png` e
   `parte1_times.csv`.

3. **Parte 2 — interfaccia grafica**

       python parte2_gui.py

   Apre la GUI: scegli un `.bmp` in toni di grigio dal filesystem,
   imposta F (lato blocco) e d (soglia di taglio, 0 ≤ d ≤ 2F−2),
   premi "Esegui compressione". Sotto i parametri appaiono MSE, PSNR,
   numero di coefficienti tenuti per blocco e tempo di esecuzione.
   È disponibile anche il salvataggio del risultato.

4. **Esperimenti riproducibili (per la relazione)**

       python parte2_esperimenti.py
       # oppure, su una propria immagine:
       python parte2_esperimenti.py mia_immagine.bmp

   Salva grafici e CSV nella cartella `esperimenti/`.
