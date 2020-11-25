import argparse
import os
from pathlib import Path
import json
from functools import partial
import numpy as np
import random
from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.utils.tensorboard
import logging
import lantern
from lantern.functional import starcompose
from lantern import set_seeds, worker_init
from datastream import Datastream

from {{cookiecutter.package_name}} import datastream, architecture, metrics


def evaluate(config):
    device = torch.device("cuda" if config["use_cuda"] else "cpu")

    model = architecture.Model().to(device)

    if Path("model").exists():
        print("Loading model checkpoint")
        model.load_state_dict(torch.load("model/model.pt"))

    evaluate_data_loaders = {
        f"evaluate_{name}": datastream.data_loader(
            batch_size=config["eval_batch_size"],
            num_workers=config["n_workers"],
            collate_fn=tuple,
        )
        for name, datastream in datastream.evaluate_datastreams().items()
    }

    tensorboard_logger = torch.utils.tensorboard.SummaryWriter()
    evaluate_metrics = {
        name: lantern.Metrics(
            name=name,
            tensorboard_logger=tensorboard_logger,
            metrics=metrics.evaluate_metrics(),
        )
        for name in evaluate_data_loaders.keys()
    }

    with lantern.module_eval(model), torch.no_grad():
        for name, data_loader in evaluate_data_loaders.items():
            for examples in tqdm(data_loader, desc=name, leave=False):
                predictions = model.predictions(
                    architecture.FeatureBatch.from_examples(examples)
                )
                loss = predictions.loss(examples)
                evaluate_metrics[name].update_(examples, predictions.cpu(), loss.cpu())
            evaluate_metrics[name].log_().print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--n_workers", default=2, type=int)

    try:
        __IPYTHON__
        args = parser.parse_known_args()[0]
    except NameError:
        args = parser.parse_args()

    config = vars(args)
    config.update(
        seed=1,
        use_cuda=torch.cuda.is_available(),
        run_id=os.getenv("RUN_ID"),
    )

    Path("config.json").write_text(json.dumps(config))

    evaluate(config)
