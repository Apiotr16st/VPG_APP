import cv2
import numpy as np


def draw_outlined_text(frame, text, origin, font_scale, color=(255, 255, 255), thickness=1):
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 3)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)



def format_bpm_text(y_state, green_state, grgb_state, sample_count, config):
    if y_state.current_bpm is None and green_state.current_bpm is None:
        return f"Zbieram bufor {min(sample_count, config.buffer_size)}/{config.buffer_size}"

    y_v = "--" if y_state.current_bpm is None else f"{y_state.current_bpm:.1f}"
    g_v = "--" if green_state.current_bpm is None else f"{green_state.current_bpm:.1f}"
    grgb_v = "--" if grgb_state.current_bpm is None else f"{grgb_state.current_bpm:.1f}"
    return f"BPM -> Y: {y_v} | G: {g_v} | GRGB: {grgb_v}"

def format_average_bpm_text(y_state, green_state, grgb_state, config):
    y_v = "--" if y_state.rolling_bpm_average is None else f"{y_state.rolling_bpm_average:.1f}"
    g_v = "--" if green_state.rolling_bpm_average is None else f"{green_state.rolling_bpm_average:.1f}"
    grgb_v = "--" if grgb_state.rolling_bpm_average is None else f"{grgb_state.rolling_bpm_average:.1f}"
    return f"Srednia -> Y: {y_v} | G: {g_v} | GRGB: {grgb_v}"

def format_eight_second_bpm_text(y_state, green_state, grgb_state, config):
    y_v = "--" if y_state.eight_second_bpm is None else f"{y_state.eight_second_bpm:.1f}"
    g_v = "--" if green_state.eight_second_bpm is None else f"{green_state.eight_second_bpm:.1f}"
    grgb_v = "--" if grgb_state.eight_second_bpm is None else f"{grgb_state.eight_second_bpm:.1f}"
    return f"Ostatnie {config.eight_second_window:.0f}s -> Y: {y_v} | G: {g_v} | GRGB: {grgb_v}"

def format_calc_time_text(y_state, green_state, grgb_state):
    y_v = "--" if y_state.current_calc_time_ms is None else f"{y_state.current_calc_time_ms:.1f}"
    g_v = "--" if green_state.current_calc_time_ms is None else f"{green_state.current_calc_time_ms:.1f}"
    grgb_v = "--" if grgb_state.current_calc_time_ms is None else f"{grgb_state.current_calc_time_ms:.1f}"
    return f"Czas [ms] -> Y: {y_v} | G: {g_v} | GRGB: {grgb_v}"

def draw_bpm_plot(frame, y_bpm, g_bpm, grgb_bpm, config, rect):
    if not y_bpm and not g_bpm and not grgb_bpm:
        return

    x0, y0, plot_w, plot_h = rect
    x1 = x0 + plot_w
    y1 = y0 + plot_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (180, 180, 180), 1)

    draw_outlined_text(frame, "Wykres tetna [BPM]", (x0 + 8, y0 + 22), 0.55)
    draw_outlined_text(frame, "Y", (x1 - 98, y0 + 22), 0.5)
    draw_outlined_text(frame, "G", (x1 - 68, y0 + 22), 0.5)
    draw_outlined_text(frame, "GRGB", (x1 - 42, y0 + 22), 0.5)

    recent_values = list(y_bpm[-config.bpm_history_limit:]) + list(g_bpm[-config.bpm_history_limit:]) + list(grgb_bpm[-config.bpm_history_limit:])
    plot_min, plot_max = config.bpm_plot_min, config.bpm_plot_max
    if recent_values:
        plot_min = min(plot_min, np.floor(min(recent_values) / 10.0) * 10.0)
        plot_max = max(plot_max, np.ceil(max(recent_values) / 10.0) * 10.0)

    start = int(np.ceil(plot_min / 20.0) * 20)
    stop = int(np.floor(plot_max / 20.0) * 20)
    marks = [int(plot_min), int(plot_max)] if start > stop else list(range(start, stop + 1, 20))

    for bpm_mark in marks:
        if bpm_mark < plot_min or bpm_mark > plot_max: continue
        y = int(y1 - ((bpm_mark - plot_min) / (plot_max - plot_min)) * (plot_h - 35) - 8)
        cv2.line(frame, (x0 + 45, y), (x1 - 8, y), (70, 70, 70), 1)
        draw_outlined_text(frame, str(bpm_mark), (x0 + 8, y + 5), 0.4)

    left, right, top, bottom = x0 + 45, x1 - 8, y0 + 30, y1 - 8

    for values_source, color in (
        (y_bpm, (0, 255, 255)),        # Zółty (Y)
        (g_bpm, (0, 255, 0)),          # Zielony (G)
        (grgb_bpm, (255, 255, 0))      # Cyjanowy (GRGB)
    ):
        values = np.asarray(values_source[-config.bpm_history_limit:], dtype=float)
        values = np.clip(values, plot_min, plot_max)
        if len(values) < 2: continue
        xs = np.linspace(left, right, len(values)).astype(int)
        ys = (bottom - ((values - plot_min) / (plot_max - plot_min)) * (bottom - top)).astype(int)
        points = np.column_stack((xs, ys)).astype(np.int32)
        cv2.polylines(frame, [points], False, color, 2)
        cv2.circle(frame, tuple(points[-1]), 4, color, -1)

def draw_raw_roi_plot(frame, y_samples, g_samples, grgb_samples, config, rect):
    x0, y0, plot_w, plot_h = rect
    x1, y1 = x0 + plot_w, y0 + plot_h
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (180, 180, 180), 1)

    draw_outlined_text(frame, "Srednia ROI przed filtrem", (x0 + 8, y0 + 22), 0.52)
    
    left, right, top, bottom = x0 + 45, x1 - 8, y0 + 30, y1 - 8
    mid_y = int((top + bottom) / 2)
    cv2.line(frame, (left, mid_y), (right, mid_y), (70, 70, 70), 1)

    sample_limit = max(config.buffer_size, int(config.eight_second_window * 30))
    for samples, color in ((y_samples, (0, 255, 255)), (g_samples, (0, 255, 0)), (grgb_samples, (255, 255, 0))):
        values = np.asarray(samples[-sample_limit:], dtype=float)
        if len(values) < 2: continue
        values = values - np.mean(values)
        scale = np.percentile(np.abs(values), 95)
        if scale <= 1e-8: continue
        values = np.clip(values / scale, -1.0, 1.0)
        xs = np.linspace(left, right, len(values)).astype(int)
        ys = (mid_y - values * ((bottom - top) * 0.45)).astype(int)
        points = np.column_stack((xs, ys)).astype(np.int32)
        cv2.polylines(frame, [points], False, color, 2)

def draw_status_overlay(frame, y_state, green_state, grgb_state, sample_count, config):
    dashboard = create_dashboard_frame(frame)
    panel_x = frame.shape[1] + 20

    draw_outlined_text(dashboard, format_bpm_text(y_state, green_state, grgb_state, sample_count, config), (panel_x, 35), 0.65, thickness=2)
    draw_outlined_text(dashboard, format_average_bpm_text(y_state, green_state, grgb_state, config), (panel_x, 70), 0.55, thickness=2)
    draw_outlined_text(dashboard, format_eight_second_bpm_text(y_state, green_state, grgb_state, config), (panel_x, 105), 0.55, thickness=2)
    draw_outlined_text(dashboard, format_calc_time_text(y_state, green_state, grgb_state), (panel_x, 140), 0.55, thickness=2)
    
    draw_bpm_plot(dashboard, y_state.bpm_history, green_state.bpm_history, grgb_state.bpm_history, config, (panel_x, 180, 460, 210))
    draw_raw_roi_plot(dashboard, y_state.samples, green_state.samples, grgb_state.samples, config, (panel_x, 410, 460, 160))
    return dashboard

def create_dashboard_frame(frame):
    frame_h, frame_w = frame.shape[:2]
    
    sidebar_w = 550
    dashboard_h = max(frame_h, 590)
    dashboard_w = frame_w + sidebar_w
    
    dashboard = np.full((dashboard_h, dashboard_w, 3), 24, dtype=np.uint8)
    dashboard[:frame_h, :frame_w] = frame
    cv2.line(dashboard, (frame_w, 0), (frame_w, dashboard_h), (80, 80, 80), 1)
    return dashboard