import os
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

from vpg.config import default_config
from vpg.face_roi import get_forehead_roi
from vpg.heart_rate import ChannelState, update_channel_state
from vpg.overlay import draw_status_overlay
from vpg.reporting import save_final_report


def create_face_landmarker(model_path):
    if not os.path.exists(model_path):
        raise SystemExit(
            "Brakuje modelu 'face_landmarker.task'. Pobierz model Face Landmarker "
            f"i zapisz go w katalogu projektu: {os.path.dirname(model_path)}"
        )

    return vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )


def print_startup_message():
    print("=====================================================")
    print("Kamera uruchomiona. Zbieram sygnal VPG...")
    print("Siedz nieruchomo przez ok. 15-20 sekund i patrz w kamere.")
    print("Aby zakonczyc pomiar i zapisac pliki, wcisnij klawisz 'q'.")
    print("=====================================================")

def main():
    config = default_config()
    face_landmarker = create_face_landmarker(config.face_landmarker_model)
    cap = cv2.VideoCapture(0)

    y_state = ChannelState()
    green_state = ChannelState()
    grgb_state = ChannelState() # Nowy kanał z artykułu
    sample_timestamps = []
    smoothed_roi = None
    start_time = time.time()

    print_startup_message()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_time = time.time()
            timestamp_ms = int((frame_time - start_time) * 1000)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = face_landmarker.detect_for_video(mp_image, timestamp_ms)

            if results.face_landmarks:
                smoothed_roi = process_face_frame(
                    frame,
                    results.face_landmarks[0],
                    smoothed_roi,
                    frame_time,
                    y_state,
                    green_state,
                    grgb_state, # Przekazujemy trzeci stan
                    sample_timestamps,
                    config,
                )

            dashboard_frame = draw_status_overlay(
                frame,
                y_state,
                green_state,
                grgb_state,
                len(y_state.samples),
                config,
            )
            display_frame = resize_for_display(
                dashboard_frame,
                config.display_scale,
                config.display_max_width,
                config.display_max_height,
            )
            cv2.imshow("VPG Prototyp - Kamera na zywo", display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        elapsed_time = time.time() - start_time
        cap.release()
        face_landmarker.close()
        cv2.destroyAllWindows()

    save_final_report(
        y_state.samples,
        green_state.samples,
        grgb_state.samples, # Zapis raportu z uwzględnieniem GRGB
        sample_timestamps,
        elapsed_time,
        config,
    )

def process_face_frame(
    frame,
    face_landmarks,
    smoothed_roi,
    frame_time,
    y_state,
    green_state,
    grgb_state,
    sample_timestamps,
    config,
):
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
        smoothed_roi = config.roi_smoothing_alpha * roi_now + (1 - config.roi_smoothing_alpha) * smoothed_roi

    fh_x, fh_y, fh_w, fh_h = [int(v) for v in smoothed_roi]
    roi_color = frame[fh_y:fh_y + fh_h, fh_x:fh_x + fh_w]

    if roi_color.size > 0:
        roi_yuv = cv2.cvtColor(roi_color, cv2.COLOR_BGR2YUV)
        
        # Ochrona przed dzieleniem przez zero dla ciemnych pikseli
        mean_b = max(1e-6, float(np.mean(roi_color[:, :, 0])))
        mean_g = float(np.mean(roi_color[:, :, 1]))
        mean_r = max(1e-6, float(np.mean(roi_color[:, :, 2])))
        mean_y = float(np.mean(roi_yuv[:, :, 0]))

        y_state.samples.append(mean_y)
        green_state.samples.append(mean_g)
        # Kalkulacja w oparciu o artykuł (G/R + G/B)
        grgb_state.samples.append((mean_g / mean_r) + (mean_g / mean_b)) 
        sample_timestamps.append(frame_time)

        update_channel_state(y_state, sample_timestamps, config)
        update_channel_state(green_state, sample_timestamps, config)
        update_channel_state(grgb_state, sample_timestamps, config, is_grgb=True)

    cv2.rectangle(frame, (sx, sy), (sx + sw, sy + sh), (255, 0, 0), 2)
    cv2.rectangle(frame, (fh_x, fh_y), (fh_x + fh_w, fh_y + fh_h), (0, 255, 0), 2)
    return smoothed_roi

def resize_for_display(frame, scale, max_width, max_height):
    frame_h, frame_w = frame.shape[:2]
    fit_scale = min(max_width / frame_w, max_height / frame_h)
    effective_scale = min(scale, fit_scale)

    if effective_scale == 1.0:
        return frame

    return cv2.resize(
        frame,
        None,
        fx=effective_scale,
        fy=effective_scale,
        interpolation=cv2.INTER_LINEAR,
    )


if __name__ == "__main__":
    main()
