from pathlib import Path

import pytest


@pytest.fixture
def sample_audio_path():
    path = Path(__file__).parent.parent / "data/one_shot_percussive_sounds/1/183.wav"
    return path
