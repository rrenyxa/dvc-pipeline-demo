from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def resolve_class_names(dataset: Any) -> list[str]:
    """Resolve class names from a dataset or nested dataset wrapper.

    Args:
        dataset: Dataset instance, optionally wrapped by ``Subset``.

    Returns:
        Ordered class names from the first object exposing ``classes``.

    Raises:
        AttributeError: If no ``classes`` attribute exists in the dataset chain.

    Examples:
        >>> class Dataset:
        ...     classes = ["cats", "dogs"]
        >>> resolve_class_names(Dataset())
        ['cats', 'dogs']
    """
    current = dataset
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        classes = getattr(current, "classes", None)
        if classes is not None:
            return [str(class_name) for class_name in classes]
        current = getattr(current, "dataset", None)
    raise AttributeError("Dataset chain does not expose a 'classes' attribute.")


def create_confusion_tensor(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Build a confusion matrix using a vectorised bincount.

    Args:
        targets: Ground-truth class indices with any shape.
        predictions: Predicted class indices with the same number of elements.
        num_classes: Number of classes.

    Returns:
        Integer confusion matrix with shape ``(num_classes, num_classes)``.

    Raises:
        ValueError: If inputs have mismatched sizes or invalid ``num_classes``.
    """
    if num_classes <= 0:
        raise ValueError("num_classes must be positive.")

    flat_targets = targets.reshape(-1).to(dtype=torch.int64, device="cpu")
    flat_predictions = predictions.reshape(-1).to(dtype=torch.int64, device="cpu")
    if flat_targets.numel() != flat_predictions.numel():
        raise ValueError("targets and predictions must contain the same number of elements.")
    if flat_targets.numel() == 0:
        return torch.zeros((num_classes, num_classes), dtype=torch.int64)

    valid = (
        (0 <= flat_targets)
        & (flat_targets < num_classes)
        & (0 <= flat_predictions)
        & (flat_predictions < num_classes)
    )
    encoded = flat_targets[valid] * num_classes + flat_predictions[valid]
    return torch.bincount(encoded, minlength=num_classes**2).reshape(num_classes, num_classes)


def classification_metrics_from_confusion(confusion: torch.Tensor) -> dict[str, float]:
    """Compute aggregate classification metrics from a confusion matrix.

    Args:
        confusion: Confusion matrix where rows are true labels and columns are predictions.

    Returns:
        Flat metrics dictionary suitable for DVC and MLflow logging.
    """
    matrix = confusion.to(dtype=torch.float64, device="cpu")
    true_positive = matrix.diag()
    support = matrix.sum(dim=1)
    predicted = matrix.sum(dim=0)

    precision = torch.where(predicted > 0, true_positive / predicted, torch.zeros_like(predicted))
    recall = torch.where(support > 0, true_positive / support, torch.zeros_like(support))
    f1 = torch.where(
        precision + recall > 0,
        2 * precision * recall / (precision + recall),
        torch.zeros_like(precision),
    )

    total = support.sum()
    weights = torch.where(total > 0, support / total, torch.zeros_like(support))
    accuracy = true_positive.sum() / total if total > 0 else torch.tensor(0.0)

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(precision.mean()) if precision.numel() else 0.0,
        "macro_recall": float(recall.mean()) if recall.numel() else 0.0,
        "macro_f1": float(f1.mean()) if f1.numel() else 0.0,
        "weighted_precision": float((precision * weights).sum()),
        "weighted_recall": float((recall * weights).sum()),
        "weighted_f1": float((f1 * weights).sum()),
    }


def classification_report_from_confusion(
    confusion: torch.Tensor,
    class_names: list[str],
) -> dict[str, Any]:
    """Create aggregate and per-class metrics from a confusion matrix.

    Args:
        confusion: Confusion matrix where rows are true labels and columns are predictions.
        class_names: Ordered class names matching matrix rows and columns.

    Returns:
        JSON-serialisable classification report.

    Raises:
        ValueError: If matrix shape and class names length are inconsistent.
    """
    _validate_confusion_inputs(confusion=confusion, class_names=class_names)
    matrix = confusion.to(dtype=torch.float64, device="cpu")
    true_positive = matrix.diag()
    support = matrix.sum(dim=1)
    predicted = matrix.sum(dim=0)

    precision = torch.where(predicted > 0, true_positive / predicted, torch.zeros_like(predicted))
    recall = torch.where(support > 0, true_positive / support, torch.zeros_like(support))
    f1 = torch.where(
        precision + recall > 0,
        2 * precision * recall / (precision + recall),
        torch.zeros_like(precision),
    )

    return {
        "summary": classification_metrics_from_confusion(confusion),
        "per_class": {
            class_name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index].item()),
            }
            for index, class_name in enumerate(class_names)
        },
    }


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader[Any],
    device: str,
    loss_fn: nn.Module,
    show_progress: bool = True,
) -> tuple[dict[str, float], torch.Tensor]:
    """Evaluate a classifier and build its confusion matrix.

    Args:
        model: PyTorch model returning class logits.
        dataloader: Evaluation dataloader yielding ``(inputs, targets)``.
        device: Device used for inference.
        loss_fn: Loss function used to compute average loss.
        show_progress: Whether to display a tqdm progress bar.

    Returns:
        Tuple of flat scalar metrics and the integer confusion matrix.

    Raises:
        AttributeError: If the dataloader dataset has no class names.
    """
    model.eval()
    total_loss = 0.0
    total_examples = 0
    class_names = resolve_class_names(dataloader.dataset)
    num_classes = len(class_names)
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    batches: Iterable[tuple[torch.Tensor, torch.Tensor]]
    batches = tqdm(dataloader, desc="Evaluating", leave=False) if show_progress else dataloader

    with torch.inference_mode():
        for inputs, targets in batches:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            loss = loss_fn(logits, targets)
            predictions = logits.argmax(dim=1)

            batch_size = targets.size(0)
            total_examples += batch_size
            total_loss += loss.item() * batch_size
            confusion += create_confusion_tensor(targets, predictions, num_classes)

    metrics = classification_metrics_from_confusion(confusion)
    metrics["loss"] = total_loss / total_examples if total_examples else 0.0
    return metrics, confusion.cpu()


def save_metrics_json(metrics: dict[str, Any], output_path: str | Path) -> Path:
    """Save metrics as deterministic UTF-8 JSON.

    Args:
        metrics: JSON-serialisable metric payload.
        output_path: Destination JSON path.

    Returns:
        Path to the saved JSON file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")
    return path


def create_confusion_matrix(
    confusion: torch.Tensor,
    class_names: list[str],
    artifact_dir: Path | None,
    image_name: str = "confusion_matrix.png",
) -> Path | None:
    """Create and optionally persist a confusion-matrix image.

    Args:
        confusion: Integer confusion matrix with shape ``(n_classes, n_classes)``.
        class_names: Ordered class labels for both axes.
        artifact_dir: Directory where the image is saved; ``None`` skips saving.
        image_name: Output image file name.

    Returns:
        Saved image path when ``artifact_dir`` is provided; otherwise ``None``.

    Raises:
        ValueError: If matrix shape and class names length are inconsistent.
    """
    _validate_confusion_inputs(confusion=confusion, class_names=class_names)
    matrix = confusion.to(device="cpu").numpy()
    figure_width = max(6.0, len(class_names) * 1.2)
    figure_height = max(5.0, len(class_names) * 1.0)

    fig, ax = plt.subplots(figsize=(figure_width, figure_height))
    try:
        image = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
        ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")
        ax.set_title("Confusion Matrix")

        threshold = matrix.max() / 2 if matrix.size else 0
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                ax.text(
                    column_index,
                    row_index,
                    f"{int(value)}",
                    ha="center",
                    va="center",
                    color="white" if value > threshold else "black",
                )

        fig.tight_layout()
        if artifact_dir is None:
            return None

        artifact_dir.mkdir(parents=True, exist_ok=True)
        output_path = artifact_dir / image_name
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
        return output_path
    finally:
        plt.close(fig)


def _validate_confusion_inputs(confusion: torch.Tensor, class_names: list[str]) -> None:
    """Validate confusion matrix and class-name compatibility.

    Args:
        confusion: Confusion matrix to validate.
        class_names: Ordered class labels.

    Returns:
        None.

    Raises:
        ValueError: If matrix dimensions are invalid.
    """
    if confusion.ndim != 2 or confusion.shape[0] != confusion.shape[1]:
        raise ValueError("confusion must be a square 2D tensor.")
    if confusion.shape[0] != len(class_names):
        raise ValueError("class_names length must match confusion matrix dimensions.")


_resolve_class_names = resolve_class_names
create_confusoin_matrix = create_confusion_matrix
