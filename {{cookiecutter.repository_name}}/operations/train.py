import argparse
import os
import json
from functools import partial
from pathlib import Path
import torch
import torch.utils.tensorboard
import lantern

from {{cookiecutter.package_name}} import (
    datastream,
    Model,
    metrics,
    log_examples,
    tools,
)


def train(config):
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if config["cuda"] else "cpu")
    lantern.set_seeds(config["seed"])

    model = Model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    if Path("model").exists():
        print("Loading model checkpoint")
        model.load_state_dict(torch.load("model/model.pt"))
        optimizer.load_state_dict(torch.load("model/optimizer.pt"))
        lantern.set_learning_rate(optimizer, config["learning_rate"])

    train_data_loader = (
        datastream.TrainDatastream()
        .map(model.StandardizedImage.from_example)
        .data_loader(
            batch_size=config["batch_size"],
            n_batches_per_epoch=config["n_batches_per_epoch"],
            collate_fn=tools.unzip,
            num_workers=config["n_workers"],
            worker_init_fn=lantern.worker_init_fn(config["seed"]),
            persistent_workers=(config["n_workers"] >= 1),
        )
    )

    evaluate_datastreams = datastream.evaluate_datastreams()
    evaluate_data_loaders = {
        f"evaluate_{name}": (
            evaluate_datastreams[name]
            .map(model.StandardizedImage.from_example)
            .data_loader(
                batch_size=config["eval_batch_size"],
                collate_fn=tools.unzip,
                num_workers=config["n_workers"],
            )
        )
        for name in ["mini_train", "early_stopping"]
    }

    tensorboard_logger = torch.utils.tensorboard.SummaryWriter(log_dir="tb")
    early_stopping = lantern.EarlyStopping(tensorboard_logger=tensorboard_logger)
    train_metrics = metrics.train_metrics()

    for epoch in lantern.Epochs(config["max_epochs"]):

        for standardized_images, examples in lantern.ProgressBar(
            train_data_loader, "train", train_metrics
        ):
            with lantern.module_train(model), torch.enable_grad():
                predictions = model.predictions(standardized_images)
                loss = predictions.loss(examples)
                loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_metrics["loss"].update_(loss)
            train_metrics["pairs"].update_(predictions, examples)

            for metric in train_metrics.values():
                metric.log_dict(tensorboard_logger, "train", epoch)

        print(lantern.MetricTable("train", train_metrics))
        log_examples(tensorboard_logger, "train", epoch, predictions, examples)

        evaluate_metrics = {
            name: metrics.evaluate_metrics() for name in evaluate_data_loaders
        }

        for dataset_name, data_loader in evaluate_data_loaders.items():
            for standardized_images, examples in lantern.ProgressBar(
                data_loader, dataset_name
            ):
                with lantern.module_eval(model):
                    predictions = model.predictions(standardized_images)

                evaluate_metrics[dataset_name]["pairs"].update_(predictions, examples)

            for metric in evaluate_metrics[dataset_name].values():
                metric.log_dict(tensorboard_logger, dataset_name, epoch)

            print(lantern.MetricTable(dataset_name, evaluate_metrics[dataset_name]))
            log_examples(tensorboard_logger, dataset_name, epoch, predictions, examples)

        early_stopping = early_stopping.score(
            evaluate_metrics["evaluate_early_stopping"]["pairs"].compute()["accuracy"]
        )
        if early_stopping.scores_since_improvement == 0:
            torch.save(model.state_dict(), "model.pt")
            torch.save(optimizer.state_dict(), "optimizer.pt")
        elif early_stopping.scores_since_improvement > config["patience"]:
            break
        early_stopping.log(epoch).print()

        tensorboard_logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--max_epochs", type=int, default=20)
    parser.add_argument("--n_batches_per_epoch", type=int, default=50)
    parser.add_argument("--patience", type=float, default=10)
    parser.add_argument("--n_workers", type=int, default=2)
    parser.add_argument("--cuda", type=bool, choices=[True, False], default=True)
    args = parser.parse_args()

    config = vars(args)
    config.update(
        seed=31415,
        run_id=os.getenv("RUN_ID"),
    )

    Path("config.json").write_text(json.dumps(config))

    train(config)
