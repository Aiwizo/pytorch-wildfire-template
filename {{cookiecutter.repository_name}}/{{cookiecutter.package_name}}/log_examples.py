import numpy as np


def log_examples(logger, name, epoch, predictions, examples):
    logger.add_images(
        f"{name}/predictions",
        np.stack(
            [
                np.array(
                    predictions[index].representation(examples[index]),
                    dtype=np.float32,
                )
                / 255
                for index in range(min(5, len(predictions)))
            ]
        ),
        epoch,
        dataformats="NHWC",
    )
