import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def test(plik_csv, fps, min_bpm=55.0, max_bpm=200.0, order=3):
    # 1. Wczytanie surowych danych ( kanał G - indeks 1)
    dane = np.loadtxt(plik_csv, delimiter=',', skiprows=1)
    g_raw = dane[:, 1]
    g_detrend = signal.detrend(g_raw) # Usuwamy liniowy trend przed analizą
    
    nyq = fps / 2.0
    low_cut = min_bpm / 60.0
    high_cut = max_bpm / 60.0
    b, a = signal.butter(order, [low_cut, high_cut], btype="bandpass", fs=fps)
    
    # 3. Przepuszczamy sygnał przez filtr
    g_filt = signal.filtfilt(b, a, g_detrend)
    
    # === WYKRES 1: CHARAKTERYSTYKA FILTRA 
    w, h = signal.freqz(b, a, worN=2000, fs=fps)
    freqs_bpm = w * 60.0 # Konwersja Hz na BPM dla czytelności
    
    plt.figure(figsize=(14, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(freqs_bpm, np.abs(h), 'b')
    plt.axvline(min_bpm, color='k', linestyle='--', label=f'Dolne odcięcie ({min_bpm} BPM)')
    plt.axvline(max_bpm, color='k', linestyle='--', label=f'Górne odcięcie ({max_bpm} BPM)')
    plt.title("Charakterystyka Amplitudowa Filtra Butterwortha")
    plt.xlabel("Częstotliwość [BPM]")
    plt.ylabel("Wzmocnienie")
    plt.xlim(0, 300)
    plt.grid(True)
    plt.legend()
    
    # === WYKRES 2 i 3: SYGNAŁ W DZIEDZINIE CZĘSTOTLIWOŚCI  ===
    f_surowe, Pxx_surowe = signal.welch(g_detrend, fs=fps, nperseg=len(g_detrend))
    f_filt, Pxx_filt = signal.welch(g_filt, fs=fps, nperseg=len(g_filt))
    
    plt.subplot(3, 1, 2)
    plt.plot(f_surowe * 60.0, Pxx_surowe, 'orange', label="Widmo surowe")
    plt.title("Widmo sygnału PRZED filtrem")
    plt.xlabel("Częstotliwość [BPM]")
    plt.ylabel("Moc")
    plt.xlim(0, 300)
    plt.grid(True)
    plt.legend()
    
    plt.subplot(3, 1, 3)
    plt.plot(f_filt * 60.0, Pxx_filt, 'green', label="Widmo po filtracji")
    plt.title("Widmo sygnału PO filtrze")
    plt.xlabel("Częstotliwość [BPM]")
    plt.ylabel("Moc")
    plt.xlim(0, 300)
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.show()


test("vpg_data_20260527_235236.csv", fps=30.0)  