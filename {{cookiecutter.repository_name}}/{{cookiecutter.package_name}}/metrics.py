import pandas as pd
import lantern


def train_metrics():
    return dict(
        loss=lantern.Metric().reduce(lambda state, loss: dict(loss=loss.item())),
        pairs=lantern.Metric().reduce(
            lambda state, predictions, examples: dict(
                accuracy=predictions.accuracy(examples).float().mean().item()
            )
        ),
    )


def evaluate_metrics():
    return dict(
        pairs=(
            lantern.Metric()
            .map(
                lambda predictions, examples: dict(
                    loss=predictions.loss(examples),
                    accuracy=predictions.accuracy(examples),
                )
            )
            .map(
                lambda metrics: {
                    name: value.detach().cpu() for name, value in metrics.items()
                }
            )
            .aggregate(lambda dicts: pd.concat(list(map(pd.DataFrame, dicts))))
            .map(lambda df: df.mean(axis=0).to_dict())
        ),
    )
