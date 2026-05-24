import datetime
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from vpg.heart_rate import estimate_heart_rate_autocorr


def save_final_report(y_samples, green_samples, timestamps, elapsed_time, config):
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

    np.savetxt(
        data_filename,
        np.column_stack((y_samples, green_samples)),
        delimiter=",",
        header="Mean_Y_Luminance,Mean_G_RGB",
        comments="",
    )

    analysis = _analyze_for_report(y_samples, green_samples, timestamps, actual_fps, config)
    if analysis is None:
        print("Zbyt malo danych w wymaganym pasmie czestotliwosci.")
        return

    _print_results(analysis)
    _save_plot(plot_filename, timestamps, analysis, config)

    print(f"-> ZAPISANO DANE SUROWE: {os.path.abspath(data_filename)}")
    print(f"-> ZAPISANO WYKRES: {os.path.abspath(plot_filename)}")
    plt.show()


def _analyze_for_report(y_samples, green_samples, timestamps, actual_fps, config):
    y_array = np.asarray(y_samples, dtype=float)
    green_array = np.asarray(green_samples, dtype=float)
    y_detrended = signal.detrend(y_array)
    green_detrended = signal.detrend(green_array)

    nyquist = actual_fps / 2.0
    low_cut = config.min_bpm / 60.0
    high_cut = min(config.max_bpm / 60.0, nyquist * 0.95)
    if low_cut <= 0 or high_cut <= low_cut:
        return None

    b, a = signal.butter(3, [low_cut, high_cut], btype="bandpass", fs=actual_fps)
    y_filtered = signal.filtfilt(b, a, y_detrended)
    green_filtered = signal.filtfilt(b, a, green_detrended)

    autocorr = signal.correlate(y_filtered, y_filtered, mode="full")
    autocorr = autocorr[len(autocorr) // 2:]
    max_autocorr = np.max(np.abs(autocorr))
    if max_autocorr <= 0:
        return None
    autocorr = autocorr / max_autocorr

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

    return {
        "actual_fps": actual_fps,
        "time_axis": np.array(timestamps) - timestamps[0],
        "y_centered": y_array - np.mean(y_array),
        "green_centered": green_array - np.mean(green_array),
        "y_detrended_display": y_detrended / (np.std(y_detrended) + 1e-8),
        "y_filtered_display": y_filtered / (np.std(y_filtered) + 1e-8),
        "green_detrended_display": green_detrended / (np.std(green_detrended) + 1e-8),
        "green_filtered_display": green_filtered / (np.std(green_filtered) + 1e-8),
        "valid_autocorr": valid_autocorr,
        "min_lag": min_lag,
        "max_lag": max_lag,
        "y_bpm": y_bpm,
        "green_bpm": green_bpm,
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
        print(f" ROZNICA: {abs(analysis['y_bpm'] - analysis['green_bpm']):.1f} BPM")
    print("=======================================\n")


def _save_plot(plot_filename, timestamps, analysis, config):
    plt.figure(figsize=(12, 9))

    plt.subplot(3, 1, 1)
    plt.plot(analysis["time_axis"], analysis["y_centered"], color="black", alpha=0.8, label="Y - srednia")
    plt.plot(analysis["time_axis"], analysis["green_centered"], color="green", alpha=0.8, label="G - srednia")
    plt.title("Porownanie kanalow Y i G z ROI po odjeciu sredniej")
    plt.xlabel("Czas [s]")
    plt.ylabel("Odchylenie od sredniej")
    plt.legend(loc="upper right")
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(analysis["time_axis"], analysis["y_centered"], color="gray", alpha=0.45, label="Y - srednia")
    plt.plot(analysis["time_axis"], analysis["y_detrended_display"], color="orange", alpha=0.75, label="Y detrend / std")
    plt.plot(analysis["time_axis"], analysis["y_filtered_display"], color="red", linewidth=1.4, label="Y po filtrze / std")
    plt.plot(analysis["time_axis"], analysis["green_detrended_display"], color="lime", alpha=0.55, label="G detrend / std")
    plt.plot(analysis["time_axis"], analysis["green_filtered_display"], color="blue", linewidth=1.2, alpha=0.85, label="G po filtrze / std")
    if analysis["green_bpm"] is None:
        plt.title(f"Sygnal VPG | Y: {analysis['y_bpm']:.1f} BPM | G: -- BPM")
    else:
        plt.title(f"Sygnal VPG | Y: {analysis['y_bpm']:.1f} BPM | G: {analysis['green_bpm']:.1f} BPM")
    plt.xlabel("Czas [s]")
    plt.ylabel("Odchylenie / wartosc znormalizowana")
    plt.legend(loc="upper right")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    bpm_axis = (analysis["actual_fps"] / np.arange(analysis["min_lag"], analysis["max_lag"] + 1)) * 60.0
    plt.plot(bpm_axis, analysis["valid_autocorr"], color="red", label="Autokorelacja Y")
    plt.axvline(analysis["y_bpm"], color="black", linestyle="--", label=f"Pik Y: {analysis['y_bpm']:.1f} BPM")
    plt.title("Estymacja tetna na podstawie autokorelacji")
    plt.xlabel("Tetno [BPM]")
    plt.ylabel("Znormalizowana autokorelacja")
    plt.legend(loc="upper right")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(plot_filename)
