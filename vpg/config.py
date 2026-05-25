import os
from dataclasses import dataclass


@dataclass(frozen=True)
class VpgConfig:
    base_dir: str
    face_landmarker_model: str
    display_scale: float = 2.0
    display_max_width: int = 1280
    display_max_height: int = 680
    roi_smoothing_alpha: float = 0.15

    buffer_size: int = 50
    min_bpm: float = 55.0
    max_bpm: float = 200.0

    bpm_plot_min: float = 60.0
    bpm_plot_max: float = 120.0
    rolling_bpm_buffer_size: int = 10
    eight_second_window: float = 8.0
    eight_second_update_interval: float = 1.0
    bpm_history_limit: int = 150


def default_config() -> VpgConfig:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return VpgConfig(
        base_dir=base_dir,
        face_landmarker_model=os.path.join(base_dir, "face_landmarker.task"),
    )
