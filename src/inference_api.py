from contextlib import asynccontextmanager
from io import StringIO
from pathlib import Path
from typing import List

import pandas as pd
import torch
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

try:
    from .dataset import FEATURE_COLUMNS, dataframe_to_windows, normalize_features
    from .model import Conv1DAutoencoder
    from .utils import setup_logging
except ImportError:
    from dataset import FEATURE_COLUMNS, dataframe_to_windows, normalize_features
    from model import Conv1DAutoencoder
    from utils import setup_logging


logger = setup_logging()

MODEL = None
CONFIG = None
DEVICE = None
NORMALIZATION = None
THRESHOLD = None


class VibrationData(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "sensor_data": [
                    [0.0] * 100,
                    [0.0] * 100,
                    [9.8] * 100,
                ]
            }
        }
    )

    sensor_data: List[List[float]] = Field(
        ...,
        description="Matriz 2D con forma [3, window_size]: canales X, Y, Z.",
    )


class PredictionResponse(BaseModel):
    anomaly: bool
    mse_loss: float
    threshold: float
    message: str


class BatchPredictionResponse(BaseModel):
    anomaly: bool
    threshold: float
    window_size: int
    step: int
    window_count: int
    max_mse_loss: float
    mean_mse_loss: float
    anomaly_window_ratio: float
    anomaly_windows: List[int]
    window_mse_losses: List[float]
    message: str


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_yaml_config(base_dir: Path):
    config_path = base_dir / "config" / "config.yaml"
    with config_path.open("r") as file:
        return yaml.safe_load(file)


def resolve_data_path(base_dir: Path, raw_data_path: str) -> Path:
    if raw_data_path.startswith("../"):
        return base_dir / raw_data_path[3:]
    return base_dir / raw_data_path


def load_training_normalization(base_dir: Path, config: dict):
    data_path = resolve_data_path(base_dir, config["data"]["train_path"])
    df = pd.read_csv(data_path)
    _, min_vals, max_vals = dataframe_to_windows(
        df,
        window_size=config["data"]["window_size"],
        step=config["data"]["step"],
    )
    return {
        "min_vals": min_vals.tolist(),
        "max_vals": max_vals.tolist(),
    }


def load_model_artifact():
    global MODEL, CONFIG, DEVICE, NORMALIZATION, THRESHOLD

    base_dir = get_project_root()
    CONFIG = load_yaml_config(base_dir)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MODEL = Conv1DAutoencoder(
        in_channels=CONFIG["model"]["in_channels"],
        filters=CONFIG["model"]["encoder_filters"],
        kernel_size=CONFIG["model"]["kernel_size"],
    ).to(DEVICE)

    model_path = base_dir / "models" / "autoencoder_v1.pth"
    if not model_path.exists():
        raise FileNotFoundError(f"No se encontro el modelo en {model_path}. Entrena el modelo primero.")

    artifact = torch.load(model_path, map_location=DEVICE)
    if isinstance(artifact, dict) and "model_state_dict" in artifact:
        MODEL.load_state_dict(artifact["model_state_dict"])
        NORMALIZATION = artifact.get("normalization")
        THRESHOLD = artifact.get("threshold")
    else:
        MODEL.load_state_dict(artifact)
        NORMALIZATION = None
        THRESHOLD = None

    if NORMALIZATION is None:
        logger.warning(
            "El modelo no incluye normalizacion. Se recalculara desde el CSV de entrenamiento."
        )
        NORMALIZATION = load_training_normalization(base_dir, CONFIG)

    if THRESHOLD is None:
        THRESHOLD = CONFIG.get("inference", {}).get("threshold", 0.005)

    MODEL.eval()
    logger.info(
        "Modelo listo. Dispositivo=%s | threshold=%.8f | min=%s | max=%s",
        DEVICE,
        THRESHOLD,
        NORMALIZATION["min_vals"],
        NORMALIZATION["max_vals"],
    )


def ensure_model_loaded():
    if MODEL is None or CONFIG is None or NORMALIZATION is None or THRESHOLD is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado todavia.")


def predict_windows(windows):
    ensure_model_loaded()
    tensor_data = torch.as_tensor(windows, dtype=torch.float32, device=DEVICE)

    with torch.no_grad():
        reconstruction = MODEL(tensor_data)
        mse_losses = torch.mean((tensor_data - reconstruction) ** 2, dim=(1, 2))

    return mse_losses.detach().cpu().numpy()


def build_batch_response(mse_losses) -> BatchPredictionResponse:
    anomaly_windows = [idx for idx, loss in enumerate(mse_losses) if float(loss) > THRESHOLD]
    max_mse = float(mse_losses.max())
    mean_mse = float(mse_losses.mean())
    anomaly_ratio = len(anomaly_windows) / len(mse_losses)
    inference_config = CONFIG.get("inference", {}) if CONFIG is not None else {}
    min_anomaly_ratio = inference_config.get("min_anomaly_ratio", 0.02)
    severe_threshold_multiplier = inference_config.get("severe_threshold_multiplier", 3.0)
    anomaly = (
        anomaly_ratio >= min_anomaly_ratio
        or max_mse >= float(THRESHOLD) * severe_threshold_multiplier
    )
    message = (
        "ALERTA: se han detectado ventanas con posible fallo en la via."
        if anomaly
        else "Tramo de via compatible con el perfil normal de vibraciones."
    )

    return BatchPredictionResponse(
        anomaly=anomaly,
        threshold=float(THRESHOLD),
        window_size=CONFIG["data"]["window_size"],
        step=CONFIG["data"]["step"],
        window_count=len(mse_losses),
        max_mse_loss=max_mse,
        mean_mse_loss=mean_mse,
        anomaly_window_ratio=anomaly_ratio,
        anomaly_windows=anomaly_windows,
        window_mse_losses=[float(loss) for loss in mse_losses],
        message=message,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Arrancando API de inferencia...")
    load_model_artifact()
    yield
    logger.info("Apagando API de inferencia.")


app = FastAPI(
    title="API de Deteccion de Anomalias en Vias de Tren",
    description="Inferencia sobre ventanas de acelerometro con un autoencoder 1D.",
    version="1.0",
    lifespan=lifespan,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def upload_page():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "threshold": THRESHOLD,
        "device": str(DEVICE) if DEVICE is not None else None,
    }


@app.get("/metadata")
def metadata():
    return {
        "feature_columns": FEATURE_COLUMNS,
        "window_size": CONFIG["data"]["window_size"] if CONFIG else None,
        "step": CONFIG["data"]["step"] if CONFIG else None,
        "threshold": THRESHOLD,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict_anomaly(data: VibrationData):
    ensure_model_loaded()

    expected_channels = CONFIG["model"]["in_channels"]
    expected_window = CONFIG["data"]["window_size"]

    if len(data.sensor_data) != expected_channels:
        raise HTTPException(
            status_code=400,
            detail=f"Se esperaban {expected_channels} canales, se recibieron {len(data.sensor_data)}.",
        )

    for idx, channel in enumerate(data.sensor_data):
        if len(channel) != expected_window:
            raise HTTPException(
                status_code=400,
                detail=f"El canal {idx} tiene {len(channel)} lecturas, se esperaban {expected_window}.",
            )

    raw_window = torch.tensor(data.sensor_data, dtype=torch.float32).numpy().T
    normalized_window = normalize_features(
        raw_window,
        NORMALIZATION["min_vals"],
        NORMALIZATION["max_vals"],
    ).T
    mse_loss = float(predict_windows([normalized_window])[0])
    anomaly = mse_loss > THRESHOLD
    message = (
        "ALERTA: posible fallo en la via detectado."
        if anomaly
        else "Tramo de via compatible con el perfil normal de vibraciones."
    )

    logger.info("Inferencia JSON. MSE=%.8f | Anomalia=%s", mse_loss, anomaly)

    return PredictionResponse(
        anomaly=anomaly,
        mse_loss=mse_loss,
        threshold=float(THRESHOLD),
        message=message,
    )


@app.post("/predict/csv", response_model=BatchPredictionResponse)
async def predict_csv(request: Request):
    ensure_model_loaded()

    raw_body = await request.body()
    if not raw_body:
        raise HTTPException(status_code=400, detail="El cuerpo CSV esta vacio.")

    try:
        df = pd.read_csv(StringIO(raw_body.decode("utf-8")))
        windows, _, _ = dataframe_to_windows(
            df,
            window_size=CONFIG["data"]["window_size"],
            step=CONFIG["data"]["step"],
            min_vals=NORMALIZATION["min_vals"],
            max_vals=NORMALIZATION["max_vals"],
        )
        mse_losses = predict_windows(windows)
    except UnicodeDecodeError as exc:
        raise HTTPException(status_code=400, detail="El CSV debe estar codificado en UTF-8.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Error procesando CSV de inferencia.")
        raise HTTPException(status_code=500, detail="Error interno procesando el CSV.") from exc

    response = build_batch_response(mse_losses)
    logger.info(
        "Inferencia CSV. Ventanas=%d | Max MSE=%.8f | Anomalia=%s",
        response.window_count,
        response.max_mse_loss,
        response.anomaly,
    )
    return response


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
