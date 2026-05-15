import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import yaml
from dotenv import load_dotenv
from torch.utils.data import DataLoader, random_split

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(base_dir, ".env")
load_dotenv(dotenv_path=env_path)

try:
    from .dataset import VibrationDataset, load_and_window_data
    from .model import Conv1DAutoencoder
    from .utils import setup_logging
except ImportError:
    from dataset import VibrationDataset, load_and_window_data
    from model import Conv1DAutoencoder
    from utils import setup_logging


def resolve_data_path(base_dir: Path, raw_data_path: str) -> Path:
    if raw_data_path.startswith("../"):
        return base_dir / raw_data_path[3:]
    return base_dir / raw_data_path


def load_config(base_dir: Path) -> dict:
    config_path = base_dir / "config" / "config.yaml"
    with config_path.open("r") as file:
        return yaml.safe_load(file)


def create_data_loaders(windows, config):
    dataset = VibrationDataset(windows)
    validation_split = config["training"].get("validation_split", 0.2)
    val_size = max(1, int(len(dataset) * validation_split))
    train_size = len(dataset) - val_size

    if train_size <= 0:
        raise ValueError("El conjunto de entrenamiento queda vacio. Reduce validation_split.")

    generator = torch.Generator().manual_seed(config["training"]["seed"])
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=generator,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
    )
    return train_loader, val_loader


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0

    for batch_x, _ in train_loader:
        batch_x = batch_x.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_x)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_x.size(0)

    return total_loss / len(train_loader.dataset)


def collect_reconstruction_errors(model, data_loader, device):
    model.eval()
    reconstruction_errors = []

    with torch.no_grad():
        for batch_x, _ in data_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            per_sample_mse = torch.mean((batch_x - outputs) ** 2, dim=(1, 2))
            reconstruction_errors.extend(per_sample_mse.cpu().numpy().tolist())

    return np.asarray(reconstruction_errors, dtype=np.float32)


def evaluate_loader(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for batch_x, _ in data_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_x)
            total_loss += loss.item() * batch_x.size(0)

    return total_loss / len(data_loader.dataset)


def evaluate_csv(model, csv_path, config, min_vals, max_vals, threshold, device):
    windows, _, _ = load_and_window_data(
        csv_path,
        window_size=config["data"]["window_size"],
        step=config["data"]["step"],
        min_vals=min_vals,
        max_vals=max_vals,
    )
    loader = DataLoader(
        VibrationDataset(windows),
        batch_size=config["training"]["batch_size"],
        shuffle=False,
    )
    errors = collect_reconstruction_errors(model, loader, device)
    anomaly_windows = np.where(errors > threshold)[0]
    anomaly_ratio = len(anomaly_windows) / len(errors)

    inference_config = config.get("inference", {})
    min_anomaly_ratio = inference_config.get("min_anomaly_ratio", 0.02)
    severe_threshold_multiplier = inference_config.get("severe_threshold_multiplier", 3.0)
    anomaly_detected = (
        anomaly_ratio >= min_anomaly_ratio
        or float(errors.max()) >= threshold * severe_threshold_multiplier
    )

    return {
        "mean_mse": float(errors.mean()),
        "max_mse": float(errors.max()),
        "anomaly_ratio": float(anomaly_ratio),
        "anomaly_detected": bool(anomaly_detected),
        "window_count": int(len(errors)),
        "anomaly_window_count": int(len(anomaly_windows)),
    }


def log_metrics(metrics):
    if wandb.run is not None:
        wandb.log(metrics)


def train():
    logger = setup_logging()
    logger.info("Iniciando pipeline de entrenamiento...")

    project_root = Path(__file__).resolve().parents[1]

    try:
        config = load_config(project_root)
        logger.info("Configuración cargada correctamente desde %s", project_root / "config" / "config.yaml")
    except Exception as e:
        logger.error("Error al cargar la configuración: %s", e)
        return

    torch.manual_seed(config["training"]["seed"])
    np.random.seed(config["training"]["seed"])

    try:
        wandb.init(
            project=config["logging"]["project_name"],
            name=config["logging"]["run_name"],
            config=config,
        )
        logger.info("Weights & Biases inicializado.")
    except Exception as e:
        logger.warning("No se pudo inicializar W&B. ¿Hiciste 'wandb login'? Error: %s", e)
        logger.warning("Continuando entrenamiento sin W&B tracking...")

    logger.info("Preparando datos...")
    data_path = resolve_data_path(project_root, config["data"]["train_path"])
    window_size = config["data"]["window_size"]
    step_size = config["data"]["step"]

    try:
        windows, min_vals, max_vals = load_and_window_data(
            data_path,
            window_size=window_size,
            step=step_size,
        )
        train_loader, val_loader = create_data_loaders(windows, config)
        logger.info(
            "Datos preparados. Ventanas train=%d | val=%d | batches train=%d | val=%d",
            len(train_loader.dataset),
            len(val_loader.dataset),
            len(train_loader),
            len(val_loader),
        )
    except Exception as e:
        logger.error("Error al procesar los datos: %s", e)
        return

    logger.info("Inicializando modelo Autoencoder...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Dispositivo de entrenamiento: %s", device)

    model = Conv1DAutoencoder(
        in_channels=config["model"]["in_channels"],
        filters=config["model"]["encoder_filters"],
        kernel_size=config["model"]["kernel_size"],
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    epochs = config["training"]["epochs"]
    logger.info("Comenzando entrenamiento por %d épocas...", epochs)

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = evaluate_loader(model, val_loader, criterion, device)

        logger.info(
            "Epoch [%d/%d] - train_loss: %.6f | val_loss: %.6f",
            epoch + 1,
            epochs,
            train_loss,
            val_loss,
        )

        log_metrics({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "val/loss": val_loss,
        })

    logger.info("Entrenamiento finalizado exitosamente.")

    val_errors = collect_reconstruction_errors(model, val_loader, device)
    threshold_quantile = config.get("inference", {}).get("threshold_quantile", 0.995)
    threshold = float(np.quantile(val_errors, threshold_quantile))
    logger.info(
        "Umbral calibrado con cuantíl %.3f de MSE de validación: %.8f",
        threshold_quantile,
        threshold,
    )

    metrics = {
        "inference/threshold": threshold,
        "inference/threshold_quantile": threshold_quantile,
        "val/reconstruction_mean": float(val_errors.mean()),
        "val/reconstruction_max": float(val_errors.max()),
    }

    test_normal_path = resolve_data_path(project_root, config["data"]["test_normal_path"])
    test_anomaly_path = resolve_data_path(project_root, config["data"]["test_anomaly_path"])
    test_normal_metrics = evaluate_csv(
        model,
        test_normal_path,
        config,
        min_vals,
        max_vals,
        threshold,
        device,
    )
    test_anomaly_metrics = evaluate_csv(
        model,
        test_anomaly_path,
        config,
        min_vals,
        max_vals,
        threshold,
        device,
    )

    separation_score = (
        test_anomaly_metrics["mean_mse"] / test_normal_metrics["mean_mse"]
        if test_normal_metrics["mean_mse"] > 0
        else float("inf")
    )

    metrics.update({
        "test_normal/mean_mse": test_normal_metrics["mean_mse"],
        "test_normal/max_mse": test_normal_metrics["max_mse"],
        "test_normal/anomaly_ratio": test_normal_metrics["anomaly_ratio"],
        "test_normal/anomaly_detected": int(test_normal_metrics["anomaly_detected"]),
        "test_normal/window_count": test_normal_metrics["window_count"],
        "test_normal/anomaly_window_count": test_normal_metrics["anomaly_window_count"],
        "test_anomaly/mean_mse": test_anomaly_metrics["mean_mse"],
        "test_anomaly/max_mse": test_anomaly_metrics["max_mse"],
        "test_anomaly/anomaly_ratio": test_anomaly_metrics["anomaly_ratio"],
        "test_anomaly/anomaly_detected": int(test_anomaly_metrics["anomaly_detected"]),
        "test_anomaly/window_count": test_anomaly_metrics["window_count"],
        "test_anomaly/anomaly_window_count": test_anomaly_metrics["anomaly_window_count"],
        "evaluation/separation_score": float(separation_score),
    })
    log_metrics(metrics)

    logger.info("Evaluación test normal: %s", test_normal_metrics)
    logger.info("Evaluación test anomalía: %s", test_anomaly_metrics)
    logger.info("Separation score anomaly/normal mean MSE: %.4f", separation_score)

    models_dir = project_root / "models"
    models_dir.mkdir(exist_ok=True)
    model_path = models_dir / "autoencoder_v1.pth"
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "normalization": {
            "min_vals": min_vals.tolist(),
            "max_vals": max_vals.tolist(),
        },
        "threshold": threshold,
        "threshold_quantile": threshold_quantile,
        "evaluation": {
            "test_normal": test_normal_metrics,
            "test_anomaly": test_anomaly_metrics,
            "separation_score": float(separation_score),
        },
    }
    torch.save(checkpoint, model_path)
    logger.info("Modelo guardado en disco: %s", model_path)

    if wandb.run is not None:
        try:
            logger.info("Registrando modelo como Artefacto en W&B...")
            artifact = wandb.Artifact(
                name="train-anomaly-autoencoder",
                type="model",
                description="CNN Autoencoder entrenado con datos normales",
            )
            artifact.add_file(str(model_path))
            wandb.log_artifact(artifact)
            logger.info("Artefacto registrado en W&B.")
        except Exception as e:
            logger.warning("No se pudo registrar el artefacto en W&B: %s", e)
        finally:
            wandb.finish()
            logger.info("Run de W&B finalizado.")


if __name__ == "__main__":
    train()
