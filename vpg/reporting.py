import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from vpg.heart_rate import estimate_heart_rate_autocorr


def save_final_report(y_samples, green_samples, grgb_samples, timestamps, elapsed_time, config):
    if len(y_samples) <= 100:
        print("Nagranie bylo za krotkie! Trzymaj twarz przed kamera przez minimum 10 sekund.")
        return

    signal_duration = timestamps[-1] - timestamps[0]
    if signal_duration <= 0:
        print("Nie udalo sie poprawnie wyznaczyc czasu probek sygnalu.")
        return

    actual_fps = (len(y_samples) - 1) / signal_duration
    print(f"\nZebrano {len(y_samples)} probek w czasie {elapsed_time:.2f} sekund.")
    print(f"Rzeczywisty klatkaz (FPS) wyniosl: {actual_fps:.2f} kl/s. Trwa analiza...")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    data_filename = f"vpg_data_{timestamp}.csv"
    plot_filename = f"vpg_plot_{timestamp}.png"

    # Zmiana 1: Zapisujemy do CSV wszystkie 3 kolumny
    np.savetxt(
        data_filename,
        np.column_stack((y_samples, green_samples, grgb_samples)),
        delimiter=",",
        header="Mean_Y_Luminance,Mean_G_RGB,GRGB_Value",
        comments="",
    )

    analysis = _analyze_for_report(y_samples, green_samples, grgb_samples, timestamps, actual_fps, config)
    if analysis is None:
        print("Zbyt malo danych w wymaganym pasmie czestotliwosci.")
        return

    _print_results(analysis)
    _save_plot(plot_filename, timestamps, analysis, config)

   # _save_txt_summary("wynik", timestamps, elapsed_time, analysis)

    print(f"-> ZAPISANO DANE SUROWE: {os.path.abspath(data_filename)}")
    print(f"-> ZAPISANO WYKRES: {os.path.abspath(plot_filename)}")
    plt.show()


def _analyze_for_report(y_samples, green_samples, grgb_samples, timestamps, actual_fps, config):
    y_array = np.asarray(y_samples, dtype=float)
    green_array = np.asarray(green_samples, dtype=float)
    grgb_array = np.asarray(grgb_samples, dtype=float)  # Nowy kanał
    
    y_detrended = signal.detrend(y_array)
    green_detrended = signal.detrend(green_array)
    grgb_detrended = signal.detrend(grgb_array)  # Nowy kanał

    nyquist = actual_fps / 2.0
    low_cut = config.min_bpm / 60.0
    high_cut = min(config.max_bpm / 60.0, nyquist * 0.95)
    
    if low_cut <= 0 or high_cut <= low_cut:
        return None

    # Ten sam filtr 3-rzędu nakładany na wszystkie trzy kanały
    b, a = signal.butter(3, [low_cut, high_cut], btype="bandpass", fs=actual_fps)
    y_filtered = signal.filtfilt(b, a, y_detrended)
    green_filtered = signal.filtfilt(b, a, green_detrended)
    grgb_filtered = signal.filtfilt(b, a, grgb_detrended)

    # Autokorelacja Y (tradycyjna)
    autocorr = signal.correlate(y_filtered, y_filtered, mode="full")
    autocorr = autocorr[len(autocorr) // 2:]
    max_autocorr = np.max(np.abs(autocorr))
    if max_autocorr <= 0:
        return None
    autocorr = autocorr / max_autocorr

    # Autokorelacja GRGB 
    autocorr_grgb = signal.correlate(grgb_filtered, grgb_filtered, mode="full")
    autocorr_grgb = autocorr_grgb[len(autocorr_grgb) // 2:]
    max_autocorr_grgb = np.max(np.abs(autocorr_grgb))
    if max_autocorr_grgb > 0:
        autocorr_grgb = autocorr_grgb / max_autocorr_grgb

    min_lag = max(1, int(actual_fps / (config.max_bpm / 60.0)))
    max_lag = min(len(autocorr) - 1, int(actual_fps / (config.min_bpm / 60.0)))
    if max_lag <= min_lag:
        return None

    valid_autocorr = autocorr[min_lag:max_lag + 1]
    if len(valid_autocorr) == 0:
        return None

    best_lag = _best_lag_from_autocorr(valid_autocorr, min_lag, actual_fps)
    y_bpm = (actual_fps / best_lag) * 60.0
    green_bpm = estimate_heart_rate_autocorr(green_samples, timestamps, config)
    # Wywołanie dla nowej metody używa tego samego filtra, zignorowaliśmy flagę w heart_rate.py
    grgb_bpm = estimate_heart_rate_autocorr(grgb_samples, timestamps, config, is_grgb=True) 

    # Obliczenie Power Spectrum dla GRGB
    power_spectrum_grgb = np.abs(np.fft.rfft(autocorr_grgb))**2
    freqs = np.fft.rfftfreq(len(autocorr_grgb), d=1.0/actual_fps)
    bpm_freqs = freqs * 60.0

    return {
        "actual_fps": actual_fps,
        "time_axis": np.array(timestamps) - timestamps[0],
        "y_centered": y_array - np.mean(y_array),
        "green_centered": green_array - np.mean(green_array),
        "grgb_centered": grgb_array - np.mean(grgb_array),
        "y_detrended_display": y_detrended / (np.std(y_detrended) + 1e-8),
        "y_filtered_display": y_filtered / (np.std(y_filtered) + 1e-8),
        "green_detrended_display": green_detrended / (np.std(green_detrended) + 1e-8),
        "green_filtered_display": green_filtered / (np.std(green_filtered) + 1e-8),
        "grgb_filtered_display": grgb_filtered / (np.std(grgb_filtered) + 1e-8),
        "valid_autocorr": valid_autocorr,
        "min_lag": min_lag,
        "max_lag": max_lag,
        "y_bpm": y_bpm,
        "green_bpm": green_bpm,
        "grgb_bpm": grgb_bpm,
        "ps_grgb": power_spectrum_grgb,
        "ps_freqs": bpm_freqs
    }


def _best_lag_from_autocorr(valid_autocorr, min_lag, actual_fps):
    min_peak_distance = max(1, int(actual_fps * 0.25))
    peaks, _ = signal.find_peaks(
        valid_autocorr,
        distance=min_peak_distance,
        prominence=0.03,
    )

    if len(peaks) == 0:
        return min_lag + int(np.argmax(valid_autocorr))

    peak_values = valid_autocorr[peaks]
    strong_peak_threshold = 0.75 * np.max(peak_values)
    strong_peaks = peaks[peak_values >= strong_peak_threshold]
    if len(strong_peaks) > 0:
        return min_lag + int(strong_peaks[0])
    return min_lag + int(peaks[np.argmax(peak_values)])


def _print_results(analysis):
    print("\n=======================================")
    print(f" WYNIK YUV/Y: {analysis['y_bpm']:.1f} BPM")
    if analysis["green_bpm"] is not None:
        print(f" WYNIK RGB/G: {analysis['green_bpm']:.1f} BPM")
        print(f" ROZNICA Y/G: {abs(analysis['y_bpm'] - analysis['green_bpm']):.1f} BPM")
    if analysis["grgb_bpm"] is not None:
        print(f" WYNIK GRGB: {analysis['grgb_bpm']:.1f} BPM")
    print("=======================================\n")


def _save_plot(plot_filename, timestamps, analysis, config):
    plt.figure(figsize=(12, 12)) # Zwiększony rozmiar figury na 4 wykresy

    plt.subplot(4, 1, 1)
    plt.plot(analysis["time_axis"], analysis["y_centered"], color="black", alpha=0.8, label="Y - srednia")
    plt.plot(analysis["time_axis"], analysis["green_centered"], color="green", alpha=0.8, label="G - srednia")
    plt.plot(analysis["time_axis"], analysis["grgb_centered"], color="cyan", alpha=0.8, label="GRGB - srednia")
    plt.title("Porownanie kanalow Y, G oraz GRGB z ROI po odjeciu sredniej")
    plt.xlabel("Czas [s]")
    plt.ylabel("Odchylenie od sredniej")
    plt.legend(loc="upper right")
    plt.grid(True)

    plt.subplot(4, 1, 2)
    plt.plot(analysis["time_axis"], analysis["y_centered"], color="gray", alpha=0.45, label="Y - srednia")
    plt.plot(analysis["time_axis"], analysis["y_detrended_display"], color="orange", alpha=0.75, label="Y detrend / std")
    plt.plot(analysis["time_axis"], analysis["y_filtered_display"], color="red", linewidth=1.4, label="Y po filtrze / std")
    plt.plot(analysis["time_axis"], analysis["green_detrended_display"], color="lime", alpha=0.55, label="G detrend / std")
    plt.plot(analysis["time_axis"], analysis["green_filtered_display"], color="blue", linewidth=1.2, alpha=0.85, label="G po filtrze / std")
    plt.plot(analysis["time_axis"], analysis["grgb_filtered_display"], color="magenta", linewidth=1.4, alpha=0.85, label="GRGB po filtrze / std")
    
    g_bpm_txt = "--" if analysis["green_bpm"] is None else f"{analysis['green_bpm']:.1f}"
    grgb_bpm_txt = "--" if analysis["grgb_bpm"] is None else f"{analysis['grgb_bpm']:.1f}"
    plt.title(f"Sygnal VPG | Y: {analysis['y_bpm']:.1f} BPM | G: {g_bpm_txt} BPM | GRGB: {grgb_bpm_txt} BPM")
    plt.xlabel("Czas [s]")
    plt.ylabel("Odchylenie / wartosc znormalizowana")
    plt.legend(loc="upper right")
    plt.grid(True)

    plt.subplot(4, 1, 3)
    bpm_axis = (analysis["actual_fps"] / np.arange(analysis["min_lag"], analysis["max_lag"] + 1)) * 60.0
    plt.plot(bpm_axis, analysis["valid_autocorr"], color="red", label="Autokorelacja Y")
    plt.axvline(analysis["y_bpm"], color="black", linestyle="--", label=f"Pik Y: {analysis['y_bpm']:.1f} BPM")
    plt.title("Estymacja tetna na podstawie autokorelacji (Tradycyjna dla Y)")
    plt.xlabel("Tetno [BPM]")
    plt.ylabel("Znormalizowana autokorelacja")
    plt.legend(loc="upper right")
    plt.grid(True)

    # NOWY WYKRES ZGODNY Z PUBLIKACJĄ
    plt.subplot(4, 1, 4)
    # Wyświetlamy tylko interesujący nas zakres częstotliwości zdefiniowany w configu (np. 39 - 240 BPM)
    valid_indices = (analysis["ps_freqs"] >= config.grgb_min_bpm) & (analysis["ps_freqs"] <= config.grgb_max_bpm)
    plt.plot(analysis["ps_freqs"][valid_indices], analysis["ps_grgb"][valid_indices], color="magenta", label="Widmo Mocy GRGB")
    if analysis["grgb_bpm"] is not None:
         plt.axvline(analysis["grgb_bpm"], color="black", linestyle="--", label=f"Pik GRGB: {analysis['grgb_bpm']:.1f} BPM")
    plt.title("Znormalizowane widmo mocy z autokorelacji dla metody GRGB (zgodnie z badaniami)")
    plt.xlabel("Tetno [BPM]")
    plt.ylabel("Znormalizowane widmo mocy")
    plt.legend(loc="upper right")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(plot_filename)


def _save_txt_summary(filename, timestamps, elapsed_time, analysis):
    """Generuje czytelny plik TXT do szybkiej analizy wynikow eksperymentu."""
    with open(filename, "w", encoding="utf-8") as f:
        f.write("=======================================\n")
        f.write("       RAPORT Z POMIARU VPG / rPPG     \n")
        f.write("=======================================\n\n")

        f.write("[1] PARAMETRY NAGRANIA\n")
        f.write(f"Czas trwania nagrania:     {elapsed_time:.2f} s\n")
        f.write(f"Liczba zebranych klatek:   {len(timestamps)}\n")
        f.write(f"Rzeczywisty klatkaz (FPS): {analysis['actual_fps']:.2f} kl/s\n\n")

        f.write("[2] WYNIKI ESTYMACJI TETNA (BPM)\n")
        f.write(f"Kanal Y (Luminancja):      {analysis['y_bpm']:.1f} BPM\n")

        g_bpm = analysis.get("green_bpm")
        if g_bpm is not None:
            f.write(f"Kanal G (Zielony):         {g_bpm:.1f} BPM\n")
        else:
            f.write("Kanal G (Zielony):         BRAK DANYCH\n")

        grgb_bpm = analysis.get("grgb_bpm")
        if grgb_bpm is not None:
            f.write(f"Metoda GRGB (Z artykulu):  {grgb_bpm:.1f} BPM\n")
        else:
            f.write("Metoda GRGB (Z artykulu):  BRAK DANYCH\n")

        f.write("\n[3] SZYBKA ANALIZA POROWNAWCZA\n")
        if g_bpm is not None and grgb_bpm is not None:
            diff_g_grgb = abs(g_bpm - grgb_bpm)
            f.write(f"Roznica pomiedzy natywnym G a nowym GRGB wynosi: {diff_g_grgb:.1f} BPM\n\n")
            
            f.write("Wniosek algorytmiczny:\n")
            if diff_g_grgb < 2.0:
                f.write(">>> BARDZO DOBRY WYNIK. Nowa metoda GRGB zachowuje sie spolegliwie\n")
                f.write(">>> i idealnie pokrywa sie z silnym, fizjologicznym kanalem G.\n")
            elif diff_g_grgb < 6.0:
                f.write(">>> AKCEPTOWALNY WYNIK. Algorytm GRGB odfiltrowal czesc szumow z kanalu G,\n")
                f.write(">>> co spowodowalo lekkie przesuniecie estymowanego tetna. Obie metody dzialaja stabilnie.\n")
            else:
                f.write(">>> DUZA ROZNICA. Metoda GRGB daje wyraznie inny wynik niz kanal G.\n")
                f.write(">>> Sugeruje to duza ilosc szumow oswietleniowych (np. zmiana natezenia swiatla) podczas nagrania,\n")
                f.write(">>> badz mikroruchy, z ktorymi GRGB stara sie walczyc (zgodnie z obietnica z artykulu naukowego).\n")
                
        f.write("\n=======================================\n")
        f.write("Koniec raportu.\n")