from dataclasses import astuple, dataclass


@dataclass(frozen=True, slots=True)
class VisualPipelineSettings:
    frames_per_second: float = 1.5
    probe_interval_seconds: float = 3.0
    refine_window_seconds: float = 12.0
    refine_interval_seconds: float = 1.0
    skip_nonref_frames: bool = False
    scene_threshold: float = 0.28
    minimum_box_area_ratio: float = 0.01
    minimum_vertical_box_area_ratio: float = 0.0015
    minimum_vertical_center_ratio: float = 0.0
    detector_input_size: int = 416
    minimum_japanese_characters: int = 3
    probe_analysis_width: int = 1280
    probe_maximum_crops: int = 32
    probe_accept_tall_weak: bool = True
    proposal_difference_threshold: int = 24
    proposal_minimum_area_ratio: float = 0.001
    proposal_maximum_area_ratio: float = 0.35
    proposal_padding_ratio: float = 0.08
    proposal_hold_frames: int = 12
    proposal_full_frame_hold_frames: int = 4
    proposal_analysis_width: int = 320
    proposal_maximum_regions: int = 2
    tracker_max_gap_seconds: float = 1.5
    tracker_minimum_observations: int = 1
    tracker_box_iou_threshold: float = 0.25
    tracker_text_similarity_threshold: float = 0.65
    tracker_hash_distance_threshold: int = 8

    def __post_init__(self) -> None:
        if not 1 <= self.frames_per_second <= 2:
            raise ValueError("visual sampling rate must be between one and two fps")
        if self.probe_interval_seconds <= 0:
            raise ValueError("visual probe interval must be positive")
        if self.refine_window_seconds <= 0:
            raise ValueError("visual refine window must be positive")
        if self.refine_interval_seconds <= 0:
            raise ValueError("visual refine interval must be positive")
        if self.minimum_japanese_characters <= 0:
            raise ValueError("minimum Japanese character count must be positive")

    @property
    def frame_interval_seconds(self) -> float:
        return 1 / self.frames_per_second

    def cache_identity(self) -> tuple[object, ...]:
        return astuple(self)
