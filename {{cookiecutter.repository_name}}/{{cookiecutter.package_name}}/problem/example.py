from PIL import Image, ImageDraw
import numpy as np
from lantern import FunctionalBase

from {{cookiecutter.package_name}} import problem, tools


class Example(FunctionalBase):
    image: np.ndarray
    class_name: str

    class Config:
        arbitrary_types_allowed = True

    @property
    def class_index(self):
        return problem.settings.CLASS_NAMES.index(self.class_name)

    def representation(self):
        image = Image.fromarray(self.image)
        draw = ImageDraw.Draw(image)
        tools.text_(draw, self.class_name, 10, 10)
        return np.array(image)

    @property
    def _repr_png_(self):
        return Image.fromarray(self.representation())._repr_png_

    def augment(self, augmenter):
        return self.replace(image=augmenter.augment(image=self.image))


def test_example():
    from imgaug import augmenters as iaa

    (
        Example(
            image=np.zeros((256, 256, 3), np.uint8),
            class_name=problem.settings.CLASS_NAMES[0],
        ).augment(iaa.Affine(scale=(0.9, 1.1)))
    )
