"""Signal processing package for neural voice conversion."""

from .preprocessing import (
    resample_audio,
    apply_preemphasis,
    inverse_preemphasis,
    segment_into_frames,
    voice_activity_detection,
    trim_silence,
    normalize_audio,
    main_preprocessing_pipeline
)

__all__ = [
    'resample_audio',
    'apply_preemphasis',
    'inverse_preemphasis',
    'segment_into_frames',
    'voice_activity_detection',
    'trim_silence',
    'normalize_audio',
    'main_preprocessing_pipeline'
]
