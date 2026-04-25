from __future__ import annotations

import torch
from torch import nn


class ImageClassifier(nn.Module):
    """Convolutional binary image classifier for cats-vs-dogs style datasets.

    Args:
        num_classes: Number of output classes.

    Examples:
        >>> model = ImageClassifier(num_classes=2)
        >>> logits = model(torch.randn(2, 3, 224, 224))
        >>> logits.shape
        torch.Size([2, 2])
    """

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self._conv_block(3, 64),
            self._conv_block(64, 128),
            self._conv_block(128, 256),
            self._conv_block(256, 512),
            nn.AdaptiveAvgPool2d((3, 3)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=512 * 3 * 3, out_features=num_classes),
        )

    @staticmethod
    def _conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
        """Create a convolutional block with downsampling.

        Args:
            in_channels: Number of input feature channels.
            out_channels: Number of output feature channels.

        Returns:
            Sequential convolution, activation, normalisation, and pooling block.

        Examples:
            >>> block = ImageClassifier._conv_block(3, 64)
            >>> isinstance(block, nn.Sequential)
            True
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            inputs: Image batch with shape ``(batch, 3, height, width)``.

        Returns:
            Class logits with shape ``(batch, num_classes)``.
        """
        return self.classifier(self.features(inputs))
