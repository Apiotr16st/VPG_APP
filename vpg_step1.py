import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import datetime
import os
import time

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

# --- 1. KONFIGURACJA I INICJALIZACJA ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FACE_LANDMARKER_MODEL = os.path.join(BASE_DIR, "face_landmarker.task")

if not os.path.exists(FACE_LANDMARKER_MODEL):
    raise SystemExit(
        "Brakuje modelu 'face_landmarker.task'. Pobierz model Face Landmarker "
        f"i zapisz go w katalogu projektu: {BASE_DIR}"
    )

face_landmarker = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=FACE_LANDMARKER_MODEL),
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
)

cap = cv2.VideoCapture(0)

# Parametry algorytmu
raw_signal = []    # Glowny sygnal (Y/luminancja) do obliczania tetna
raw_signal_y = []  # Dodatkowa lista dla kanalu Y
raw_signal_u = []  # Dodatkowa lista dla kanalu U
raw_signal_v = []  # Dodatkowa lista dla kanalu V
sample_timestamps = []  # Czas realnie zapisanych probek sygnalu
smoothed_roi = None
alpha = 0.15


def clamp_roi(x, y, w, h, frame_width, frame_height):
    x1 = max(0, min(frame_width - 1, int(round(x))))
    y1 = max(0, min(frame_height - 1, int(round(y))))
    x2 = max(0, min(frame_width, int(round(x + w))))
    y2 = max(0, min(frame_height, int(round(y + h))))
    return x1, y1, max(0, x2 - x1), max(0, y2 - y1)


def get_forehead_roi(face_landmarks, frame_width, frame_height):
    landmarks = getattr(face_landmarks, "landmark", face_landmarks)
    points = np.array([
        [landmark.x * frame_width, landmark.y * frame_height]
        for landmark in landmarks
    ])

    face_x_min, face_y_min = np.min(points, axis=0)
    face_x_max, face_y_max = np.max(points, axis=0)
    face_w = face_x_max - face_x_min
    face_h = face_y_max - face_y_min

    brow_points = points[[70, 63, 105, 66, 107, 336, 296, 334, 293, 300]]
    brow_center = np.mean(brow_points, axis=0)
    brow_y = float(np.mean(brow_points[:, 1]))

    forehead_top = points[10]
    forehead_span = max(8.0, brow_y - forehead_top[1])

    roi_w = 0.36 * face_w
    roi_h = min(0.36 * face_h, 1 * forehead_span)
    roi_x = brow_center[0] - roi_w / 2 + 0.03 * face_w
    roi_y = forehead_top[1] + 0.12 * forehead_span

    brow_margin = 0.22 * forehead_span
    max_roi_bottom = brow_y - brow_margin
    if roi_y + roi_h > max_roi_bottom:
        roi_h = max(6.0, max_roi_bottom - roi_y)

    face_box_top_padding = 0.18 * face_h
    roi = clamp_roi(roi_x, roi_y, roi_w, roi_h, frame_width, frame_height)
    face_box = clamp_roi(
        face_x_min,
        face_y_min - face_box_top_padding,
        face_w,
        face_h + face_box_top_padding,
        frame_width,
        frame_height,
    )
    return roi, face_box

print("=====================================================")
print("Kamera uruchomiona. Zbieram sygnał VPG...")
print("Siedź nieruchomo przez ok. 15-20 sekund i patrz w kamerę.")
print("Aby zakończyć pomiar i zapisać pliki, wciśnij klawisz 'q'.")
print("=====================================================")

start_time = time.time() 

# --- 2. GŁÓWNA PĘTLA POBIERANIA OBRAZU ---
while True:
    ret, frame = cap.read()
    if not ret:
        print("Błąd pobierania obrazu.")
        break

    frame_time = time.time()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp_ms = int((frame_time - start_time) * 1000)
    results = face_landmarker.detect_for_video(mp_image, timestamp_ms)

    if results.face_landmarks:
        face_landmarks = results.face_landmarks[0]
        frame_h, frame_w = frame.shape[:2]
        (fh_x, fh_y, fh_w, fh_h), (sx, sy, sw, sh) = get_forehead_roi(
            face_landmarks,
            frame_w,
            frame_h,
        )

        roi_now = np.array([fh_x, fh_y, fh_w, fh_h], dtype=float)
        if smoothed_roi is None:
            smoothed_roi = roi_now
        else:
            smoothed_roi = alpha * roi_now + (1 - alpha) * smoothed_roi

        fh_x, fh_y, fh_w, fh_h = [int(v) for v in smoothed_roi]

        roi_color = frame[fh_y:fh_y+fh_h, fh_x:fh_x+fh_w]
        
        if roi_color.size > 0:
            roi_yuv = cv2.cvtColor(roi_color, cv2.COLOR_BGR2YUV)
            mean_y, mean_u, mean_v = np.mean(roi_yuv, axis=(0, 1))
            
            raw_signal_y.append(mean_y)
            raw_signal_u.append(mean_u)
            raw_signal_v.append(mean_v)
            
            raw_signal.append(mean_y)
            sample_timestamps.append(frame_time)
        
        cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), (255, 0, 0), 2)
        cv2.rectangle(frame, (fh_x, fh_y), (fh_x + fh_w, fh_y + fh_h), (0, 255, 0), 2)

    cv2.imshow('VPG Prototyp - Kamera na zywo', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

end_time = time.time() 
elapsed_time = end_time - start_time

cap.release()
face_landmarker.close()
cv2.destroyAllWindows()

if len(raw_signal) > 100:
    signal_duration = sample_timestamps[-1] - sample_timestamps[0]
    if signal_duration <= 0:
        print("Nie udalo sie poprawnie wyznaczyc czasu probek sygnalu.")
        raise SystemExit(1)

    actual_fps = (len(raw_signal) - 1) / signal_duration
    
    print(f"\nZebrano {len(raw_signal)} próbek w czasie {elapsed_time:.2f} sekund.")
    print(f"Rzeczywisty klatkaż (FPS) wyniósł: {actual_fps:.2f} kl/s. Trwa analiza...")
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    data_filename = f"vpg_data_{timestamp}.csv"
    plot_filename = f"vpg_plot_{timestamp}.png"
    
    np.savetxt(data_filename, raw_signal, delimiter=",", header="Mean_Y_Luminance", comments='')
    
    raw_signal_array = np.asarray(raw_signal, dtype=float)
    detrended_signal = signal.detrend(raw_signal_array)
    
    min_bpm = 55.0
    max_bpm = 200.0

    # Pasmo tetna: zostawiamy lekki zapas ponizej 60 BPM, ale bez wpuszczania
    # bardzo wolnych zmian od oswietlenia i ruchu.
    b, a = signal.butter(3, [min_bpm / 60.0, max_bpm / 60.0], btype='bandpass', fs=actual_fps)
    filtered_signal = signal.filtfilt(b, a, detrended_signal)
    time_axis = np.array(sample_timestamps) - sample_timestamps[0]
    raw_signal_centered = raw_signal_array - np.mean(raw_signal_array)
    detrended_display = detrended_signal / (np.std(detrended_signal) + 1e-8)
    filtered_display = filtered_signal / (np.std(filtered_signal) + 1e-8)
    
    autocorr = signal.correlate(filtered_signal, filtered_signal, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]
    autocorr = autocorr / np.max(np.abs(autocorr))
    
    min_lag = max(1, int(actual_fps / (max_bpm / 60.0)))
    max_lag = min(len(autocorr) - 1, int(actual_fps / (min_bpm / 60.0)))
    valid_autocorr = autocorr[min_lag:max_lag + 1]
    
    if len(valid_autocorr) > 0:
        min_peak_distance = max(1, int(actual_fps * 0.25))
        peaks, _ = signal.find_peaks(
            valid_autocorr,
            distance=min_peak_distance,
            prominence=0.03,
        )

        if len(peaks) > 0:
            peak_values = valid_autocorr[peaks]
            strong_peak_threshold = 0.75 * np.max(peak_values)
            strong_peaks = peaks[peak_values >= strong_peak_threshold]
            best_lag = min_lag + int(strong_peaks[0])
        else:
            best_lag = min_lag + int(np.argmax(valid_autocorr))

        heart_rate_hz = actual_fps / best_lag
        heart_rate_bpm = heart_rate_hz * 60.0
        
        print(f"\n=======================================")
        print(f" WYNIK: ESTYMOWANE TĘTNO = {heart_rate_bpm:.1f} BPM")
        print(f"=======================================\n")
        
        plt.figure(figsize=(12, 9))
        
        # Wykres 1: Analiza skladowych YUV po odjeciu sredniej
        plt.subplot(3, 1, 1)
        y_centered = np.asarray(raw_signal_y) - np.mean(raw_signal_y)
        u_centered = np.asarray(raw_signal_u) - np.mean(raw_signal_u)
        v_centered = np.asarray(raw_signal_v) - np.mean(raw_signal_v)
        plt.plot(time_axis, y_centered, color='black', alpha=0.8, label='Y - srednia')
        plt.plot(time_axis, u_centered, color='blue', alpha=0.6, label='U - srednia')
        plt.plot(time_axis, v_centered, color='red', alpha=0.6, label='V - srednia')
        plt.title("Zmiany kanalow YUV z ROI po odjeciu sredniej")
        plt.xlabel("Czas [s]")
        plt.ylabel("Odchylenie od sredniej")
        plt.legend(loc="upper right")
        plt.grid(True)
        
        # Wykres 2: Kanal Y w skali, w ktorej widac puls
        plt.subplot(3, 1, 2)
        plt.plot(time_axis, raw_signal_centered, color='gray', alpha=0.45, label='Y - srednia')
        plt.plot(time_axis, detrended_display, color='green', alpha=0.85, label='Y detrend / std')
        plt.plot(time_axis, filtered_display, color='red', linewidth=1.4, label='Y po filtrze / std')
        plt.title(f"Sygnal VPG (kanal Y) | Ostateczny wynik: {heart_rate_bpm:.1f} BPM")
        plt.xlabel("Czas [s]")
        plt.ylabel("Odchylenie / wartosc znormalizowana")
        plt.legend(loc="upper right")
        plt.grid(True)
        
        # Wykres 3: Autokorelacja i wybrane tetno
        plt.subplot(3, 1, 3)
        bpm_axis = (actual_fps / np.arange(min_lag, max_lag + 1)) * 60.0
        plt.plot(bpm_axis, valid_autocorr, color='red', label='Autokorelacja')
        plt.axvline(heart_rate_bpm, color='black', linestyle='--', label=f'Pik: {heart_rate_bpm:.1f} BPM')
        plt.title("Estymacja tetna na podstawie autokorelacji")
        plt.xlabel("Tetno [BPM]")
        plt.ylabel("Znormalizowana autokorelacja")
        plt.legend(loc="upper right")
        plt.grid(True)
        
        plt.tight_layout()
        
        plt.savefig(plot_filename)
        print(f"-> ZAPISANO DANE SUROWE: {os.path.abspath(data_filename)}")
        print(f"-> ZAPISANO WYKRES: {os.path.abspath(plot_filename)}")
        
        plt.show()
    else:
        print("Zbyt mało danych w wymaganym paśmie częstotliwości.")
else:
    print("Nagranie było za krótkie! Trzymaj wciśniętą twarz przed kamerą przez minimum 10 sekund.")
