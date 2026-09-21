from __future__ import annotations

import numpy as np
import pytest

from constrained_omrs.paper_animation import PaperAnimationConfig, _sample_index


def test_paper_animation_config_rejects_nonpositive_speed():
    with pytest.raises(ValueError):
        PaperAnimationConfig(playback_speed=0.0)


def test_paper_animation_sample_index_is_right_continuous():
    time = np.array([0.0, 0.1, 0.2, 0.3])
    assert _sample_index(time, 0.0) == 0
    assert _sample_index(time, 0.19) == 1
    assert _sample_index(time, 0.2) == 2
    assert _sample_index(time, 10.0) == 3
