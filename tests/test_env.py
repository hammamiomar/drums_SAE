import pytest


def test_audio_file_path(sample_audio_path):
    assert sample_audio_path.exists()


def test_torchcodec_import():
    try:
        import torchcodec
    except ImportError as e:
        pytest.fail(
            f"ok you prob got ffmpeg linkage issue.. activate the .env thing Error:{e}"
        )
