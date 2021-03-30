import argparse
import os
from pathlib import Path
import json
from tqdm import tqdm
import torch
import torch.utils.tensorboard
import lantern

from {{cookiecutter.package_name}} import datastream, architecture, metrics, tools


def evaluate(config):
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if config["use_cuda"] else "cpu")

    model = architecture.Model().to(device)

    if Path("model").exists():
        print("Loading model checkpoint")
        model.load_state_dict(torch.load("model/model.pt"))

    evaluate_data_loaders = {
        f"evaluate_{name}": (
            datastream.map(architecture.StandardizedImage.from_example).data_loader(
                batch_size=config["eval_batch_size"],
                collate_fn=tools.unzip,
                num_workers=config["n_workers"],
            )
        )
        for name, datastream in datastream.evaluate_datastreams().items()
        if "mini" not in name
    }

    tensorboard_logger = torch.utils.tensorboard.SummaryWriter()
    evaluate_metrics = {
        name: metrics.evaluate_metrics() for name in evaluate_data_loaders
    }

    for name, data_loader in evaluate_data_loaders.items():
        for examples, standardized_images in tqdm(data_loader, desc=name, leave=False):
            with lantern.module_eval(model):
                predictions = model.predictions(standardized_images)
                loss = predictions.loss(examples)

            evaluate_metrics[name]["loss"].update_(loss)
            evaluate_metrics[name]["accuracy"].update_(examples, predictions)

        for metric_name, metric in evaluate_metrics[name].items():
            metric.log(tensorboard_logger, name, metric_name)

        print(lantern.MetricTable(name, evaluate_metrics[name]))

    tensorboard_logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_batch_size", type=int, default=64)
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

    evaluate(config)
