import time
from dataclasses import dataclass, field

import numpy as np
from scipy import signal


@dataclass
class ChannelState:
    samples: list[float] = field(default_factory=list)
    current_bpm: float | None = None
    current_calc_time_ms: float | None = None
    eight_second_bpm: float | None = None
    last_eight_second_update_time: float | None = None
    rolling_bpm_buffer: list[float] = field(default_factory=list)
    rolling_bpm_average: float | None = None
    samples_since_rolling_average_update: int = 0
    bpm_history: list[float] = field(default_factory=list)


def estimate_heart_rate_autocorr(values, timestamps, config):
    if len(values) < config.buffer_size or len(timestamps) < config.buffer_size:
        return None

    signal_duration = timestamps[-1] - timestamps[0]
    if signal_duration <= 0:
        return None

    actual_fps = (len(values) - 1) / signal_duration
    nyquist = actual_fps / 2.0
    low_cut = config.min_bpm / 60.0
    high_cut = config.max_bpm / 60.0
    if low_cut <= 0 or low_cut >= nyquist:
        return None

    high_cut = min(high_cut, nyquist * 0.95)
    if high_cut <= low_cut:
        return None

    filtered_signal = preprocess_for_heart_rate(values, actual_fps, config)
    if filtered_signal is None:
        return None

    autocorr = signal.correlate(filtered_signal, filtered_signal, mode="full")
    autocorr = autocorr[len(autocorr) // 2:]
    max_abs_autocorr = np.max(np.abs(autocorr))
    if max_abs_autocorr <= 0:
        return None

    autocorr = autocorr / max_abs_autocorr
    min_lag = max(1, int(actual_fps / (config.max_bpm / 60.0)))
    max_lag = min(len(autocorr) - 1, int(actual_fps / (config.min_bpm / 60.0)))
    if max_lag <= min_lag:
        return None

    valid_autocorr = autocorr[min_lag:max_lag + 1]
    if len(valid_autocorr) == 0:
        return None

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
        if len(strong_peaks) > 0:
            best_lag = min_lag + int(strong_peaks[0])
        else:
            best_lag = min_lag + int(peaks[np.argmax(peak_values)])
    else:
        best_lag = min_lag + int(np.argmax(valid_autocorr))

    heart_rate_hz = actual_fps / best_lag
    return heart_rate_hz * 60.0


def preprocess_for_heart_rate(values, actual_fps, config):
    nyquist = actual_fps / 2.0
    low_cut = config.min_bpm / 60.0
    high_cut = min(config.max_bpm / 60.0, nyquist * 0.95)
    if low_cut <= 0 or low_cut >= nyquist or high_cut <= low_cut:
        return None

    raw_signal_array = np.asarray(values, dtype=float)
    detrended_signal = signal.detrend(raw_signal_array)
    b, a = signal.butter(3, [low_cut, high_cut], btype="bandpass", fs=actual_fps)
    return signal.filtfilt(b, a, detrended_signal)


def update_channel_state(channel_state, timestamps, config):
    if len(channel_state.samples) < config.buffer_size:
        return

    sample_buffer = channel_state.samples[-config.buffer_size:]
    timestamp_buffer = timestamps[-config.buffer_size:]
    calc_start = time.perf_counter()
    estimated_bpm = estimate_heart_rate_autocorr(sample_buffer, timestamp_buffer, config)
    channel_state.current_calc_time_ms = (time.perf_counter() - calc_start) * 1000.0
    update_eight_second_bpm(channel_state, timestamps, config)

    if estimated_bpm is None:
        return

    channel_state.current_bpm = estimated_bpm
    channel_state.rolling_bpm_buffer.append(estimated_bpm)
    if len(channel_state.rolling_bpm_buffer) > config.rolling_bpm_buffer_size:
        channel_state.rolling_bpm_buffer = channel_state.rolling_bpm_buffer[-config.rolling_bpm_buffer_size:]
    channel_state.samples_since_rolling_average_update += 1
    if (
        len(channel_state.rolling_bpm_buffer) == config.rolling_bpm_buffer_size
        and channel_state.samples_since_rolling_average_update >= config.rolling_bpm_buffer_size
    ):
        channel_state.rolling_bpm_average = float(np.mean(channel_state.rolling_bpm_buffer))
        channel_state.samples_since_rolling_average_update = 0

    channel_state.bpm_history.append(estimated_bpm)
    if len(channel_state.bpm_history) > config.bpm_history_limit:
        channel_state.bpm_history = channel_state.bpm_history[-config.bpm_history_limit:]


def update_eight_second_bpm(channel_state, timestamps, config):
    if not timestamps:
        return

    current_time = timestamps[-1]
    if (
        channel_state.last_eight_second_update_time is not None
        and current_time - channel_state.last_eight_second_update_time < config.eight_second_update_interval
    ):
        return

    window_start = current_time - config.eight_second_window
    start_index = 0
    for index, timestamp in enumerate(timestamps):
        if timestamp >= window_start:
            start_index = index
            break

    window_samples = channel_state.samples[start_index:]
    window_timestamps = timestamps[start_index:]
    if len(window_samples) < config.buffer_size:
        return

    estimated_bpm = estimate_heart_rate_autocorr(window_samples, window_timestamps, config)
    if estimated_bpm is not None:
        channel_state.eight_second_bpm = estimated_bpm
        channel_state.last_eight_second_update_time = current_time
