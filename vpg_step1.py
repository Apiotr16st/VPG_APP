import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, fft
import datetime
import os
import time

# --- 1. KONFIGURACJA I INICJALIZACJA ---
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

# Parametry algorytmu
raw_signal = []    # Główny sygnał (zielony) do obliczania tętna
raw_signal_r = []  # Dodatkowa lista dla koloru czerwonego
raw_signal_g = []  # Dodatkowa lista dla koloru zielonego
raw_signal_b = []  # Dodatkowa lista dla koloru niebieskiego
smoothed_box = None
alpha = 0.15  

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

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) > 0:
        (x, y, w, h) = faces[0]
        
        if smoothed_box is None:
            smoothed_box = [x, y, w, h]
        else:
            smoothed_box[0] = alpha * x + (1 - alpha) * smoothed_box[0]
            smoothed_box[1] = alpha * y + (1 - alpha) * smoothed_box[1]
            smoothed_box[2] = alpha * w + (1 - alpha) * smoothed_box[2]
            smoothed_box[3] = alpha * h + (1 - alpha) * smoothed_box[3]
            
        sx, sy, sw, sh = [int(v) for v in smoothed_box]

        fh_w = int(0.4 * sw)
        fh_h = int(0.16 * sh)
        fh_x = sx + int(0.3 * sw)
        fh_y = sy + int(0.13 * sh)

        roi_color = frame[fh_y:fh_y+fh_h, fh_x:fh_x+fh_w]
        
        if roi_color.size > 0:
            mean_b = np.mean(roi_color[:, :, 0])
            mean_g = np.mean(roi_color[:, :, 1])
            mean_r = np.mean(roi_color[:, :, 2])
            
            raw_signal_b.append(mean_b)
            raw_signal_g.append(mean_g)
            raw_signal_r.append(mean_r)
            
            raw_signal.append(mean_g)
        
        cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), (255, 0, 0), 2)
        cv2.rectangle(frame, (fh_x, fh_y), (fh_x + fh_w, fh_y + fh_h), (0, 255, 0), 2)

    cv2.imshow('VPG Prototyp - Kamera na zywo', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

end_time = time.time() 
elapsed_time = end_time - start_time

cap.release()
cv2.destroyAllWindows()

if len(raw_signal) > 100:  
    actual_fps = len(raw_signal) / elapsed_time 
    
    print(f"\nZebrano {len(raw_signal)} próbek w czasie {elapsed_time:.2f} sekund.")
    print(f"Rzeczywisty klatkaż (FPS) wyniósł: {actual_fps:.2f} kl/s. Trwa analiza...")
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    data_filename = f"vpg_data_{timestamp}.csv"
    plot_filename = f"vpg_plot_{timestamp}.png"
    
    np.savetxt(data_filename, raw_signal, delimiter=",", header="Mean_Green_Intensity", comments='')
    
    detrended_signal = signal.detrend(raw_signal)
    
    #GRANICA FILTRA: od 1.0 Hz (odcina wszystko poniżej 60 BPM)
    b, a = signal.butter(3, [1.0, 4.0], btype='bandpass', fs=actual_fps)
    filtered_signal = signal.filtfilt(b, a, detrended_signal)
    
    N = len(filtered_signal)
    yf = np.abs(fft.rfft(filtered_signal))
    xf = fft.rfftfreq(N, 1/actual_fps)
    
    valid_idx = np.where((xf >= 1.0) & (xf <= 4.0))
    valid_xf = xf[valid_idx]
    valid_yf = yf[valid_idx]
    
    if len(valid_yf) > 0:
        max_idx = np.argmax(valid_yf)
        heart_rate_hz = valid_xf[max_idx]
        heart_rate_bpm = heart_rate_hz * 60.0
        
        print(f"\n=======================================")
        print(f" WYNIK: ESTYMOWANE TĘTNO = {heart_rate_bpm:.1f} BPM")
        print(f"=======================================\n")
        
        plt.figure(figsize=(12, 9))
        
        # Wykres 1: Analiza składowych RGB
        plt.subplot(3, 1, 1)
        plt.plot(raw_signal_r, color='red', alpha=0.7, label='Czerwony (Red)')
        plt.plot(raw_signal_g, color='green', alpha=0.7, label='Zielony (Green)')
        plt.plot(raw_signal_b, color='blue', alpha=0.7, label='Niebieski (Blue)')
        plt.title("Porównanie surowych sygnałów RGB z czoła")
        plt.ylabel("Jasność pikseli")
        plt.legend(loc="upper right")
        plt.grid(True)
        
        # Wykres 2: Detrending dla kanału zielonego
        plt.subplot(3, 1, 2)
        plt.plot(raw_signal, color='gray', alpha=0.5, label='Surowy kanał zielony')
        plt.plot(detrended_signal, color='green', label='Po usunięciu trendu')
        plt.title(f"Sygnał VPG (Kanał Zielony) | Ostateczny wynik: {heart_rate_bpm:.1f} BPM")
        plt.xlabel("Numer próbki (klatka wideo)")
        plt.ylabel("Znormalizowana jasność")
        plt.legend(loc="upper right")
        plt.grid(True)
        
        # Wykres 3: Ostateczne tętno
        plt.subplot(3, 1, 3)
        plt.plot(filtered_signal, color='red', label='Sygnał tętna')
        plt.title("Wyczyszczony puls (Filtrowanie od 1.0 do 4.0 Hz)")
        plt.xlabel("Numer próbki (klatka wideo)")
        plt.ylabel("Amplituda")
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