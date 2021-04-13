from datastream import Datastream, samplers

from {{cookiecutter.package_name}} import datastream


def evaluate_datastreams():
    return {
        split_name: Datastream(dataset, samplers.SequentialSampler(len(dataset)))
        for split_name, dataset in datastream.datasets().items()
    }
