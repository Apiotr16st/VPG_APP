import cv2
import numpy as np


def draw_outlined_text(frame, text, origin, font_scale, color=(255, 255, 255), thickness=1):
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 3)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)


def format_bpm_text(y_state, green_state, sample_count, config):
    if y_state.current_bpm is None and green_state.current_bpm is None:
        return f"Tetno: zbieram bufor {min(sample_count, config.buffer_size)}/{config.buffer_size}"

    y_value = "--" if y_state.current_bpm is None else f"{y_state.current_bpm:.1f}"
    green_value = "--" if green_state.current_bpm is None else f"{green_state.current_bpm:.1f}"
    return f"Tetno Y: {y_value} BPM | G: {green_value} BPM"


def format_average_bpm_text(y_state, green_state, config):
    y_average = "--" if y_state.rolling_bpm_average is None else f"{y_state.rolling_bpm_average:.1f}"
    green_average = "--" if green_state.rolling_bpm_average is None else f"{green_state.rolling_bpm_average:.1f}"
    y_count = len(y_state.rolling_bpm_buffer)
    green_count = len(green_state.rolling_bpm_buffer)
    return (
        f"Srednia {config.rolling_bpm_buffer_size} pomiarow "
        f"Y: {y_average} ({y_count}) | G: {green_average} ({green_count})"
    )


def format_eight_second_bpm_text(y_state, green_state, config):
    y_value = "--" if y_state.eight_second_bpm is None else f"{y_state.eight_second_bpm:.1f}"
    green_value = "--" if green_state.eight_second_bpm is None else f"{green_state.eight_second_bpm:.1f}"
    return f"Ostatnie {config.eight_second_window:.0f}s Y: {y_value} BPM | G: {green_value} BPM"


def format_calc_time_text(y_state, green_state):
    if y_state.current_calc_time_ms is None and green_state.current_calc_time_ms is None:
        return "Czas obliczen: -- ms"

    y_calc = "--" if y_state.current_calc_time_ms is None else f"{y_state.current_calc_time_ms:.2f}"
    green_calc = "--" if green_state.current_calc_time_ms is None else f"{green_state.current_calc_time_ms:.2f}"
    return f"Czas Y: {y_calc} ms | G: {green_calc} ms"


def draw_bpm_plot(frame, y_bpm_values, green_bpm_values, config, rect):
    if not y_bpm_values and not green_bpm_values:
        return

    x0, y0, plot_w, plot_h = rect
    if plot_w < 120 or plot_h < 120:
        return

    x1 = x0 + plot_w
    y1 = y0 + plot_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    cv2.rectangle(frame, (x0, y0), (x1, y1), (180, 180, 180), 1)
    draw_outlined_text(frame, "Wykres tetna [BPM]", (x0 + 8, y0 + 22), 0.55)
    draw_outlined_text(frame, "Y", (x1 - 68, y0 + 22), 0.5)
    draw_outlined_text(frame, "G", (x1 - 38, y0 + 22), 0.5)

    plot_min, plot_max = _dynamic_bpm_plot_range(y_bpm_values, green_bpm_values, config)

    for bpm_mark in _bpm_grid_marks(plot_min, plot_max):
        if bpm_mark < plot_min or bpm_mark > plot_max:
            continue
        y = int(y1 - ((bpm_mark - plot_min) / (plot_max - plot_min)) * (plot_h - 35) - 8)
        cv2.line(frame, (x0 + 45, y), (x1 - 8, y), (70, 70, 70), 1)
        draw_outlined_text(frame, str(bpm_mark), (x0 + 8, y + 5), 0.4)

    left = x0 + 45
    right = x1 - 8
    top = y0 + 30
    bottom = y1 - 8

    for values_source, color in (
        (y_bpm_values, (0, 255, 255)),
        (green_bpm_values, (0, 255, 0)),
    ):
        values = np.asarray(values_source[-config.bpm_history_limit:], dtype=float)
        values = np.clip(values, plot_min, plot_max)
        if len(values) < 2:
            continue

        xs = np.linspace(left, right, len(values)).astype(int)
        ys = (bottom - ((values - plot_min) / (plot_max - plot_min)) * (bottom - top)).astype(int)
        points = np.column_stack((xs, ys)).astype(np.int32)
        cv2.polylines(frame, [points], False, color, 2)
        cv2.circle(frame, tuple(points[-1]), 4, color, -1)


def draw_raw_roi_plot(frame, y_samples, green_samples, config, rect):
    if len(y_samples) < 2 and len(green_samples) < 2:
        return

    x0, y0, plot_w, plot_h = rect
    if plot_w < 120 or plot_h < 120:
        return

    x1 = x0 + plot_w
    y1 = y0 + plot_h
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    cv2.rectangle(frame, (x0, y0), (x1, y1), (180, 180, 180), 1)
    draw_outlined_text(frame, "Srednia ROI przed filtrem", (x0 + 8, y0 + 22), 0.52)
    draw_outlined_text(frame, "Y", (x1 - 68, y0 + 22), 0.5)
    draw_outlined_text(frame, "G", (x1 - 38, y0 + 22), 0.5)

    left = x0 + 45
    right = x1 - 8
    top = y0 + 30
    bottom = y1 - 8
    mid_y = int((top + bottom) / 2)
    cv2.line(frame, (left, mid_y), (right, mid_y), (70, 70, 70), 1)

    sample_limit = max(config.buffer_size, int(config.eight_second_window * 30))
    for samples, color in (
        (y_samples, (0, 255, 255)),
        (green_samples, (0, 255, 0)),
    ):
        values = np.asarray(samples[-sample_limit:], dtype=float)
        if len(values) < 2:
            continue

        values = values - np.mean(values)
        scale = np.percentile(np.abs(values), 95)
        if scale <= 1e-8:
            continue

        values = np.clip(values / scale, -1.0, 1.0)
        xs = np.linspace(left, right, len(values)).astype(int)
        ys = (mid_y - values * ((bottom - top) * 0.45)).astype(int)
        points = np.column_stack((xs, ys)).astype(np.int32)
        cv2.polylines(frame, [points], False, color, 2)


def _dynamic_bpm_plot_range(y_bpm_values, green_bpm_values, config):
    plot_min = config.bpm_plot_min
    plot_max = config.bpm_plot_max
    recent_values = list(y_bpm_values[-config.bpm_history_limit:]) + list(green_bpm_values[-config.bpm_history_limit:])
    if recent_values:
        plot_min = min(plot_min, np.floor(min(recent_values) / 10.0) * 10.0)
        plot_max = max(plot_max, np.ceil(max(recent_values) / 10.0) * 10.0)
    return plot_min, plot_max


def _bpm_grid_marks(plot_min, plot_max):
    start = int(np.ceil(plot_min / 20.0) * 20)
    stop = int(np.floor(plot_max / 20.0) * 20)
    if start > stop:
        return [int(plot_min), int(plot_max)]
    return list(range(start, stop + 1, 20))


def draw_status_overlay(frame, y_state, green_state, sample_count, config):
    dashboard = create_dashboard_frame(frame)
    frame_w = frame.shape[1]
    panel_x = frame_w + 20

    bpm_text = format_bpm_text(y_state, green_state, sample_count, config)
    average_text = format_average_bpm_text(y_state, green_state, config)
    eight_second_text = format_eight_second_bpm_text(y_state, green_state, config)
    calc_text = format_calc_time_text(y_state, green_state)

    draw_outlined_text(dashboard, bpm_text, (panel_x, 35), 0.72, thickness=2)
    draw_outlined_text(dashboard, average_text, (panel_x, 70), 0.62, thickness=2)
    draw_outlined_text(dashboard, eight_second_text, (panel_x, 105), 0.62, thickness=2)
    draw_outlined_text(dashboard, calc_text, (panel_x, 140), 0.62, thickness=2)
    draw_bpm_plot(dashboard, y_state.bpm_history, green_state.bpm_history, config, (panel_x, 180, 460, 210))
    draw_raw_roi_plot(dashboard, y_state.samples, green_state.samples, config, (panel_x, 410, 460, 160))
    return dashboard


def create_dashboard_frame(frame):
    frame_h, frame_w = frame.shape[:2]
    sidebar_w = 520
    dashboard_h = max(frame_h, 590)
    dashboard_w = frame_w + sidebar_w
    dashboard = np.full((dashboard_h, dashboard_w, 3), 24, dtype=np.uint8)
    dashboard[:frame_h, :frame_w] = frame
    cv2.line(dashboard, (frame_w, 0), (frame_w, dashboard_h), (80, 80, 80), 1)
    return dashboard
