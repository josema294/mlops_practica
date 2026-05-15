from pathlib import Path

import pandas as pd

import src.inference_api as api


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _predict_csv(path):
    df = pd.read_csv(PROJECT_ROOT / path)
    windows, _, _ = api.dataframe_to_windows(
        df,
        window_size=api.CONFIG["data"]["window_size"],
        step=api.CONFIG["data"]["step"],
        min_vals=api.NORMALIZATION["min_vals"],
        max_vals=api.NORMALIZATION["max_vals"],
    )
    losses = api.predict_windows(windows)
    return api.build_batch_response(losses)


def test_model_artifact_loads_with_threshold_and_normalization():
    api.load_model_artifact()

    assert api.MODEL is not None
    assert api.THRESHOLD > 0
    assert set(api.NORMALIZATION) == {"min_vals", "max_vals"}
    assert len(api.NORMALIZATION["min_vals"]) == 3
    assert len(api.NORMALIZATION["max_vals"]) == 3


def test_normal_vibrations_do_not_trigger_alert():
    api.load_model_artifact()

    response = _predict_csv("tests/datatest/test_vibrations_normal.csv")

    assert response.anomaly is False
    assert response.window_count > 0
    assert response.max_mse_loss < response.threshold * 3


def test_anomalous_vibrations_trigger_alert():
    api.load_model_artifact()

    response = _predict_csv("tests/datatest/test_vibrations_anomaly.csv")

    assert response.anomaly is True
    assert response.window_count > 0
    assert response.max_mse_loss > response.threshold * 3
    assert response.anomaly_windows
