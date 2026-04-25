import logging
import shutil
from pathlib import Path

import click

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def copy_n_images(source_dir: Path, output_dir: Path, num_images: int | None) -> None:
    for class_dir in source_dir.iterdir():
        if not class_dir.is_dir():
            continue

        logger.info(f"Processing class {class_dir.name}")
        copied_images = 0

        for image_path in class_dir.iterdir():
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            output_path = output_dir / class_dir.name / image_path.name
            output_path.parent.mkdir(parents=True, exist_ok=True)

            logger.debug(f"Copying {image_path} to {output_path}")
            shutil.copy(image_path, output_path)

            copied_images += 1
            if num_images is not None and copied_images >= num_images:
                break

        logger.info(
            f"Copied {copied_images} images for class {class_dir.name} "
            f"to {output_dir / class_dir.name}"
        )


@click.command()
@click.option(
    "--train-dir",
    type=click.Path(exists=True, path_type=Path),
    default=Path("data/raw/training_set"),
    show_default=True,
    help="Directory containing train images organized by class"
)
@click.option(
    "--test-dir",
    type=click.Path(exists=True, path_type=Path),
    default=Path("data/raw/test_set"),
    show_default=True,
    help="Directory containing test images organized by class"
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("data/preprocessed"),
    show_default=True,
    help="Directory to save preprocessed images"
)
@click.option(
    "--num-train",
    type=int,
    default=10,
    show_default=True,
    help="Number of training images to use",
)
@click.option(
    "--num-test",
    type=int,
    default=10,
    show_default=True,
    help="Number of test images to use",
)
def preprocess(
    test_dir: Path,
    train_dir: Path, 
    output_dir: Path, 
    num_train: int | None,
    num_test: int | None):
    copy_n_images(train_dir, output_dir / "train", num_train)
    copy_n_images(test_dir, output_dir / "test", num_test)


if __name__ == "__main__":
    preprocess()