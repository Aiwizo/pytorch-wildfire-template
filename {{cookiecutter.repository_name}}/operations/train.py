import argparse
import os
import json
from functools import partial
from pathlib import Path
import torch
import torch.utils.tensorboard
import lantern
from lantern import set_seeds, worker_init_fn

from {{cookiecutter.package_name}} import (
    datastream,
    architecture,
    metrics,
    log_examples,
    tools,
)


def train(config):
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if config["use_cuda"] else "cpu")
    set_seeds(config["seed"])

    model = architecture.Model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    if Path("model").exists():
        print("Loading model checkpoint")
        model.load_state_dict(torch.load("model/model.pt"))
        optimizer.load_state_dict(torch.load("model/optimizer.pt"))
        lantern.set_learning_rate(optimizer, config["learning_rate"])

    train_data_loader = (
        datastream.TrainDatastream()
        .map(architecture.StandardizedImage.from_example)
        .data_loader(
            batch_size=config["batch_size"],
            n_batches_per_epoch=config["n_batches_per_epoch"],
            collate_fn=tools.unzip,
            num_workers=config["n_workers"],
            worker_init_fn=worker_init_fn(config["seed"]),
            persistent_workers=(config["n_workers"] >= 1),
        )
    )

    evaluate_data_loaders = {
        f"evaluate_{name}": (
            datastream.map(architecture.StandardizedImage.from_example).data_loader(
                batch_size=config["eval_batch_size"],
                collate_fn=tools.unzip,
                num_workers=config["n_workers"],
            )
        )
        for name, datastream in datastream.evaluate_datastreams().items()
        if "mini" in name
    }

    tensorboard_logger = torch.utils.tensorboard.SummaryWriter(log_dir="tb")
    early_stopping = lantern.EarlyStopping(tensorboard_logger=tensorboard_logger)
    train_metrics = metrics.train_metrics()

    for epoch in lantern.Epochs(config["max_epochs"]):

        for examples, standardized_images in lantern.ProgressBar(
            train_data_loader, "train", train_metrics
        ):
            with lantern.module_train(model), torch.enable_grad():
                predictions = model.predictions(standardized_images)
                loss = predictions.loss(examples)
                loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_metrics["loss"].update_(loss)
            train_metrics["accuracy"].update_(examples, predictions)

            for name, metric in train_metrics.items():
                metric.log(tensorboard_logger, "train", name, epoch)

        print(lantern.MetricTable("train", train_metrics))
        log_examples(tensorboard_logger, "train", epoch, examples, predictions)

        evaluate_metrics = {
            name: metrics.evaluate_metrics() for name in evaluate_data_loaders
        }

        for name, data_loader in evaluate_data_loaders.items():
            for examples, standardized_images in lantern.ProgressBar(data_loader, name):
                with lantern.module_eval(model):
                    predictions = model.predictions(standardized_images)
                    loss = predictions.loss(examples)

                evaluate_metrics[name]["loss"].update_(loss)
                evaluate_metrics[name]["accuracy"].update_(examples, predictions)

            for metric_name, metric in evaluate_metrics[name].items():
                metric.log(tensorboard_logger, name, metric_name, epoch)

            print(lantern.MetricTable(name, evaluate_metrics[name]))
            log_examples(tensorboard_logger, name, epoch, examples, predictions)

        early_stopping = early_stopping.score(
            evaluate_metrics["evaluate_mini_early_stopping"]["accuracy"].compute()
        )
        if early_stopping.scores_since_improvement == 0:
            torch.save(model.state_dict(), "model.pt")
            torch.save(optimizer.state_dict(), "optimizer.pt")
        elif early_stopping.scores_since_improvement > config["patience"]:
            break
        early_stopping.log(epoch).print()

        tensorboard_logger.close()


if __name__ == "__main__":
    if os.getenv("GUILD_DEBUG") == "1" and os.getenv("GUILD_DEBUG_STARTED") != "1":
        # configure debugger to set GUILD_DEBUG to "1"
        os.environ["GUILD_DEBUG_STARTED"] = "1"
        from guild.commands.run import run

        run(
            [
                "train",
                "-y",
                "n_workers=0",
                "n_batches_per_epoch=2",
                "--debug-sourcecode=.",
            ]
        )
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--eval_batch_size", type=int, default=64)
        parser.add_argument("--learning_rate", type=float, default=5e-4)
        parser.add_argument("--max_epochs", type=int, default=20)
        parser.add_argument("--n_batches_per_epoch", default=50, type=int)
        parser.add_argument("--patience", type=float, default=10)
        parser.add_argument("--n_workers", default=2, type=int)
        args = parser.parse_args()

        config = vars(args)
        config.update(
            seed=1,
            use_cuda=torch.cuda.is_available(),
            run_id=os.getenv("RUN_ID"),
            **tools.git_status(),
        )

        Path("config.json").write_text(json.dumps(config))

        train(config)
