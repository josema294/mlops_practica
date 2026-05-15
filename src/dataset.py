import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

FEATURE_COLUMNS = ["accel_x", "accel_y", "accel_z"]


def validate_vibration_dataframe(df):
    """Valida que el DataFrame contiene las columnas esperadas del acelerometro."""
    missing_columns = [col for col in FEATURE_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Faltan columnas obligatorias: {missing_columns}")

    if len(df) == 0:
        raise ValueError("El DataFrame de vibraciones esta vacio.")


def normalize_features(features, min_vals, max_vals):
    """Aplica normalizacion Min-Max usando estadisticos aprendidos en entrenamiento."""
    min_vals = np.asarray(min_vals, dtype=np.float32)
    max_vals = np.asarray(max_vals, dtype=np.float32)
    denominator = np.where((max_vals - min_vals) == 0, 1.0, max_vals - min_vals)
    return (features - min_vals) / denominator


def create_windows(features, window_size=100, step=50):
    """Convierte una matriz temporal (n_samples, channels) en ventanas para Conv1d."""
    if len(features) < window_size:
        raise ValueError(
            f"Se necesitan al menos {window_size} lecturas para crear una ventana; "
            f"se recibieron {len(features)}."
        )

    windows = []
    for i in range(0, len(features) - window_size + 1, step):
        windows.append(features[i:i + window_size])

    windows = np.asarray(windows, dtype=np.float32)
    return np.transpose(windows, (0, 2, 1))


def dataframe_to_windows(df, window_size=100, step=50, min_vals=None, max_vals=None):
    """Extrae columnas de acelerometro, normaliza y devuelve ventanas para inferencia."""
    validate_vibration_dataframe(df)
    features = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)

    if min_vals is None:
        min_vals = features.min(axis=0)
    if max_vals is None:
        max_vals = features.max(axis=0)

    features_norm = normalize_features(features, min_vals, max_vals)
    return create_windows(features_norm, window_size=window_size, step=step), min_vals, max_vals


def load_and_window_data(filepath, window_size=100, step=50, min_vals=None, max_vals=None):
    """Carga el CSV, normaliza (Min-Max simplificado) y extrae ventanas temporales."""
    df = pd.read_csv(filepath)
    windows, min_vals, max_vals = dataframe_to_windows(
        df,
        window_size=window_size,
        step=step,
        min_vals=min_vals,
        max_vals=max_vals,
    )
    return windows, min_vals, max_vals

class VibrationDataset(Dataset):
    """Dataset de PyTorch para las secuencias de vibración."""
    def __init__(self, data):
        self.data = torch.FloatTensor(data)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Autoencoder: el target es el input
        return self.data[idx], self.data[idx]
