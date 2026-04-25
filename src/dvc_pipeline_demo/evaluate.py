from __future__ import annotations

import logging
import os
import random
from pathlib import Path
from time import perf_counter

import click
import numpy as np
import torch
from torch import nn

from dvc_pipeline_demo.data import get_test_dataloader
from dvc_pipeline_demo.metrics import (
    create_confusion_matrix,
    evaluate_model,
    resolve_class_names,
    save_metrics_json,
)
from dvc_pipeline_demo.model_arch import ImageClassifier

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

SEED = int(os.getenv("SEED", "42"))
BATCH_SIZE = 32
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
DEFAULT_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible evaluation.

    Args:
        seed: Seed value used by all supported random number generators.

    Returns:
        None.

    Examples:
        >>> set_global_seed(42)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = False


def load_model(model_path: Path, device: str) -> nn.Module:
    """Load the image classifier checkpoint onto the requested device.

    Args:
        model_path: Path to a PyTorch ``state_dict`` checkpoint.
        device: Torch device name, e.g. ``cpu``, ``cuda``, or ``mps``.

    Returns:
        Model in evaluation mode.

    Raises:
        FileNotFoundError: If ``model_path`` does not exist.
        RuntimeError: If the checkpoint cannot be loaded into the model.

    Examples:
        >>> model = load_model(Path("models/model.pth"), "cpu")
        >>> isinstance(model, nn.Module)
        True
    """
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    model = ImageClassifier().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    return model.eval()


@click.command()
@click.option(
    "--test-dir",
    default=Path("data/preprocessed/test"),
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    show_default=True,
    help="Path to test data in ImageFolder format.",
)
@click.option(
    "-m",
    "--model-path",
    default=Path("models/model.pth"),
    type=click.Path(path_type=Path, dir_okay=False),
    show_default=True,
    help="Path to the trained model checkpoint.",
)
@click.option("-b", "--batch-size", default=BATCH_SIZE, type=int, show_default=True)
@click.option("-d", "--device", default=DEFAULT_DEVICE, type=str, show_default=True)
@click.option("-nw", "--num-workers", default=-1, type=int, show_default=True)
@click.option("-w", "--image-width", default=IMAGE_WIDTH, type=int, show_default=True)
@click.option("-h", "--image-height", default=IMAGE_HEIGHT, type=int, show_default=True)
@click.option(
    "-o",
    "--output-dir",
    default=Path("models/artifacts"),
    type=click.Path(path_type=Path),
    show_default=True,
    help="Directory for metrics and confusion-matrix artifacts.",
)
def evaluate(
    test_dir: Path,
    model_path: Path,
    batch_size: int,
    device: str,
    num_workers: int,
    image_width: int,
    image_height: int,
    output_dir: Path,
) -> None:
    """Evaluate a trained image classifier and save metrics artifacts.

    Args:
        test_dir: Directory with test images organised by class subfolders.
        model_path: Path to a saved PyTorch model ``state_dict``.
        batch_size: Number of images per inference batch.
        device: Torch device used for inference.
        num_workers: Number of dataloader workers; negative means auto.
        image_width: Resize width for input images.
        image_height: Resize height for input images.
        output_dir: Directory where evaluation artifacts are written.

    Returns:
        None.

    Examples:
        >>> # CLI: python -m dvc_pipeline_demo.evaluate --device cpu
    """
    if batch_size <= 0:
        raise click.BadParameter("batch_size must be positive.", param_hint="batch-size")

    set_global_seed(SEED)
    workers = max((os.cpu_count() or 2) - 2, 0) if num_workers < 0 else num_workers
    image_size = (image_height, image_width)

    logger.info("Loading test data from %s", test_dir)
    test_dataloader = get_test_dataloader(
        test_dir=test_dir,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=workers,
    )

    logger.info("Loading model from %s", model_path)
    model = load_model(model_path=model_path, device=device)

    started_at = perf_counter()
    metrics, confusion = evaluate_model(
        model=model,
        dataloader=test_dataloader,
        device=device,
        loss_fn=nn.CrossEntropyLoss(),
    )
    elapsed_seconds = perf_counter() - started_at
    metrics["evaluation_time_seconds"] = elapsed_seconds

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = save_metrics_json(metrics, output_dir / "metrics.json")

    confusion_path = create_confusion_matrix(
        confusion=confusion,
        class_names=resolve_class_names(test_dataloader.dataset),
        artifact_dir=output_dir,
    )

    logger.info("Metrics: %s", metrics)
    logger.info("Saved metrics to %s", metrics_path)
    if confusion_path is not None:
        logger.info("Saved confusion matrix to %s", confusion_path)


if __name__ == "__main__":
    evaluate()
