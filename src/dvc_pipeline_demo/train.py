from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

import inspect  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
from pathlib import Path  # noqa: E402
from timeit import default_timer as timer  # noqa: E402
from typing import Callable  # noqa: E402

import click  # noqa: E402
import mlflow  # noqa: E402
import mlflow.pytorch  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from mlflow.models.signature import infer_signature  # noqa: E402
from torch import Tensor, nn  # noqa: E402
from tqdm import tqdm  # noqa: E402

from dvc_pipeline_demo.data import get_train_val_dataloaders  # noqa: E402
from dvc_pipeline_demo.metrics import (  # noqa: E402
    create_confusion_matrix,
    evaluate_model,
    resolve_class_names,
)
from dvc_pipeline_demo.model_arch import ImageClassifier  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
MLFLOW_EXPERIMENT_NAME = os.environ.get("MLFLOW_EXPERIMENT_NAME", "cats-dogs")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
MLFLOW_REGISTERED_MODEL_NAME = os.environ.get(
    "MLFLOW_REGISTERED_MODEL_NAME",
    "cats-dogs-model",
)
SEED = int(os.getenv("SEED", "42"))

NUM_EPOCHS = 1
BATCH_SIZE = 32
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
LEARNING_RATE = 1e-3
VAL_SPLIT = 0.2


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs.

    Args:
        seed: Seed value used for reproducible training setup.

    Returns:
        None.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def configure_mlflow_tracking() -> None:
    """Configure MLflow tracking from environment variables.

    Args:
        None.

    Returns:
        None.
    """
    if MLFLOW_TRACKING_URI:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    logger.info("MLflow tracking URI: %s", mlflow.get_tracking_uri())
    logger.info("MLflow experiment: %s", MLFLOW_EXPERIMENT_NAME)


def train_step(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
) -> tuple[float, float]:
    """Run one training epoch.

    Args:
        model: Model to optimise.
        dataloader: Training dataloader.
        loss_fn: Objective function.
        optimizer: Parameter optimiser.
        device: Torch device name.

    Returns:
        Average loss and accuracy for the epoch.
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_examples += batch_size
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()

    return (
        total_loss / total_examples if total_examples else 0.0,
        total_correct / total_examples if total_examples else 0.0,
    )


def validation_step(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    device: str,
) -> tuple[float, float]:
    """Run one validation epoch.

    Args:
        model: Model to evaluate.
        dataloader: Validation dataloader.
        loss_fn: Objective function.
        device: Torch device name.

    Returns:
        Average loss and accuracy for the epoch.
    """
    metrics, _ = evaluate_model(
        model=model,
        dataloader=dataloader,
        device=device,
        loss_fn=loss_fn,
        show_progress=False,
    )
    return metrics["loss"], metrics["accuracy"]


def _train(
    model: nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    val_dataloader: torch.utils.data.DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    epochs: int = 5,
    metrics_logger: Callable[[dict[str, float], int], None] | None = None,
) -> dict[str, list[float]]:
    """Train a model and collect epoch metrics.

    Args:
        model: Model to train.
        train_dataloader: Training dataloader.
        val_dataloader: Validation dataloader.
        loss_fn: Objective function.
        optimizer: Parameter optimiser.
        device: Torch device name.
        epochs: Number of training epochs.
        metrics_logger: Optional callback for external metric logging.

    Returns:
        Dictionary with per-epoch loss and accuracy series.
    """
    results: dict[str, list[float]] = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    for epoch in tqdm(range(epochs), desc="Training"):
        train_loss, train_accuracy = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
        )
        val_loss, val_accuracy = validation_step(
            model=model,
            dataloader=val_dataloader,
            loss_fn=loss_fn,
            device=device,
        )
        epoch_metrics = {
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        }

        logger.info(
            "Epoch %d/%d | train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f",
            epoch + 1,
            epochs,
            train_loss,
            train_accuracy,
            val_loss,
            val_accuracy,
        )

        for metric_name, metric_value in epoch_metrics.items():
            results[metric_name].append(metric_value)
        if metrics_logger is not None:
            metrics_logger(epoch_metrics, epoch + 1)

    return results


def _log_model_to_mlflow(
    model: nn.Module,
    signature: object | None,
    input_example: np.ndarray | None,
) -> None:
    """Log a PyTorch model using the installed MLflow API shape.

    Args:
        model: CPU model to log.
        signature: Optional MLflow model signature.
        input_example: Optional sample input array.

    Returns:
        None.
    """
    log_model_kwargs: dict[str, object] = {
        "pytorch_model": model,
        "registered_model_name": MLFLOW_REGISTERED_MODEL_NAME,
    }
    artifact_parameter = (
        "name"
        if "name" in inspect.signature(mlflow.pytorch.log_model).parameters
        else "artifact_path"
    )
    log_model_kwargs[artifact_parameter] = "model"
    if signature is not None:
        log_model_kwargs["signature"] = signature
    if input_example is not None:
        log_model_kwargs["input_example"] = input_example
    if "pip_requirements" in inspect.signature(mlflow.pytorch.log_model).parameters:
        # uv environments may not include pip; skip MLflow's environment inference.
        log_model_kwargs["pip_requirements"] = []

    mlflow.pytorch.log_model(**log_model_kwargs)


@click.command()
@click.option(
    "--train-dir",
    "train_dir",
    default=Path("data/preprocessed/train"),
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    show_default=True,
    help="Path to training data in ImageFolder format.",
)
@click.option("-e", "--num-epochs", default=NUM_EPOCHS, type=int, show_default=True)
@click.option("-b", "--batch-size", default=BATCH_SIZE, type=int, show_default=True)
@click.option("-lr", "--learning-rate", default=LEARNING_RATE, type=float, show_default=True)
@click.option("-d", "--device", default=DEFAULT_DEVICE, type=str, show_default=True)
@click.option("-nw", "--num-workers", default=-1, type=int, show_default=True)
@click.option("-w", "--image-width", default=IMAGE_WIDTH, type=int, show_default=True)
@click.option("-h", "--image-height", default=IMAGE_HEIGHT, type=int, show_default=True)
@click.option("--val-split", default=VAL_SPLIT, type=float, show_default=True)
@click.option(
    "-o",
    "--output-dir",
    default=Path("models"),
    type=click.Path(path_type=Path),
    show_default=True,
)
def train(
    train_dir: Path,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
    num_workers: int,
    image_width: int,
    image_height: int,
    val_split: float,
    output_dir: Path,
) -> None:
    """Train the image classifier and log metrics/artifacts.

    Args:
        train_dir: Directory with class subfolders.
        num_epochs: Number of training epochs.
        batch_size: Number of images per batch.
        learning_rate: Adam learning rate.
        device: Torch device name.
        num_workers: Dataloader worker count; negative means automatic.
        image_width: Resize width.
        image_height: Resize height.
        val_split: Fraction of training data used for validation.
        output_dir: Directory where model and artifacts are saved.

    Returns:
        None.
    """
    if num_epochs <= 0:
        raise click.BadParameter("num_epochs must be positive.", param_hint="num-epochs")
    if batch_size <= 0:
        raise click.BadParameter("batch_size must be positive.", param_hint="batch-size")

    set_global_seed(SEED)
    resolved_workers = max((os.cpu_count() or 2) - 2, 0) if num_workers < 0 else num_workers
    image_size = (image_height, image_width)
    train_dataloader, val_dataloader = get_train_val_dataloaders(
        train_dir=train_dir,
        image_size=image_size,
        batch_size=batch_size,
        val_split=val_split,
        num_workers=resolved_workers,
    )

    signature_input_example: Tensor | None = None
    try:
        sample_inputs, _ = next(iter(val_dataloader))
        signature_input_example = sample_inputs[:1].detach().clone() if sample_inputs.numel() else None
    except StopIteration:
        logger.warning("Validation dataloader is empty; MLflow signature will be skipped.")

    class_names = resolve_class_names(val_dataloader.dataset)
    model = ImageClassifier(num_classes=len(class_names)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    total_params = sum(param.numel() for param in model.parameters())
    start_time = timer()

    configure_mlflow_tracking()
    with mlflow.start_run():
        mlflow.log_params(
            {
                "train_dir": str(train_dir),
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "device": device,
                "num_workers": resolved_workers,
                "image_width": image_width,
                "image_height": image_height,
                "val_split": val_split,
                "optimizer": optimizer.__class__.__name__,
                "loss_fn": loss_fn.__class__.__name__,
                "model": model.__class__.__name__,
                "total_parameters": total_params,
            }
        )
        mlflow.set_tags({"framework": "pytorch"})

        model_results = _train(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            device=device,
            epochs=num_epochs,
            metrics_logger=lambda metrics, epoch: mlflow.log_metrics(metrics, step=epoch),
        )
        evaluation_metrics, confusion = evaluate_model(
            model=model,
            dataloader=val_dataloader,
            device=device,
            loss_fn=loss_fn,
            show_progress=False,
        )
        mlflow.log_metrics(
            {
                "validation_loss": evaluation_metrics["loss"],
                "validation_accuracy": evaluation_metrics["accuracy"],
            },
            step=num_epochs,
        )

        artifact_dir = output_dir / "artifacts"
        confusion_path = create_confusion_matrix(
            confusion=confusion,
            class_names=class_names,
            artifact_dir=artifact_dir,
        )
        if confusion_path is not None:
            mlflow.log_artifact(str(confusion_path), artifact_path="confusion_matrix")

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "model.pth"
        logger.info("Saving model to %s", output_path)
        torch.save(model.state_dict(), output_path)
        mlflow.log_artifact(str(output_path))

        signature = None
        input_example: np.ndarray | None = None
        if signature_input_example is not None:
            example_input_cpu = signature_input_example.detach().cpu()
            model_was_training = model.training
            model.eval()
            with torch.inference_mode():
                example_output = model(example_input_cpu.to(device))
            if model_was_training:
                model.train()
            input_example = example_input_cpu.numpy()
            signature = infer_signature(input_example, example_output.detach().cpu().numpy())

        logger.info("Logging model to MLflow Model Registry as %s", MLFLOW_REGISTERED_MODEL_NAME)
        model_cpu = model.to("cpu")
        _log_model_to_mlflow(model=model_cpu, signature=signature, input_example=input_example)
        model.to(device)

        total_time = timer() - start_time
        logger.info("Total training time %.3f seconds", total_time)
        mlflow.log_metric("training_time_seconds", total_time)

        for metric_name, values in model_results.items():
            if values:
                mlflow.log_metric(f"final_{metric_name}", values[-1])


if __name__ == "__main__":
    train()
