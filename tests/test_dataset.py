import pandas as pd
import pytest

from src.dataset import FEATURE_COLUMNS, dataframe_to_windows


def test_dataframe_to_windows_returns_conv1d_shape():
    df = pd.DataFrame({
        "accel_x": range(120),
        "accel_y": range(120, 240),
        "accel_z": range(240, 360),
    })

    windows, min_vals, max_vals = dataframe_to_windows(df, window_size=100, step=50)

    assert windows.shape == (1, len(FEATURE_COLUMNS), 100)
    assert min_vals.tolist() == [0, 120, 240]
    assert max_vals.tolist() == [119, 239, 359]
    assert windows.min() >= 0
    assert windows.max() <= 1


def test_dataframe_to_windows_requires_accelerometer_columns():
    df = pd.DataFrame({
        "accel_x": [0.1, 0.2],
        "accel_y": [0.0, 0.1],
    })

    with pytest.raises(ValueError, match="Faltan columnas"):
        dataframe_to_windows(df, window_size=2, step=1)


def test_dataframe_to_windows_requires_enough_samples():
    df = pd.DataFrame({
        "accel_x": [0.1],
        "accel_y": [0.0],
        "accel_z": [9.8],
    })

    with pytest.raises(ValueError, match="al menos 100 lecturas"):
        dataframe_to_windows(df, window_size=100, step=50)
