from PIL import Image
import numpy as np
import imgaug
import torch
from lantern import FunctionalBase, Tensor, Numpy
from typing import List

from {{cookiecutter.package_name}} import settings, problem, model


resize = imgaug.augmenters.Resize(
    dict(
        height=settings.INPUT_HEIGHT,
        width=settings.INPUT_WIDTH,
    )
)


def standardize(image: Numpy) -> Tensor:
    return torch.as_tensor(image).permute(2, 0, 1).float() / 255 * 2 - 1


class StandardizedImage(FunctionalBase):
    data: Tensor

    @staticmethod
    def from_image(image: Image.Image):
        return StandardizedImage(data=standardize(resize(image=image)))

    @staticmethod
    def from_example(example: problem.Example):
        resized_example = example.augment(resize)
        return (
            StandardizedImage.from_image(resized_example.image),
            resized_example,
        )
