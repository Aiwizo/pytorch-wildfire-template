from typing import List
from pathlib import Path
from argparse import ArgumentParser
from tqdm.auto import tqdm
from CleanNB.clean_notebook import clean_notebook as clean_file


def clean_notebook(filename: str):
    notebooks_dir = Path("notebooks")
    loading_bar = tqdm(list(notebooks_dir.glob(filename)))
    for notebook_path in loading_bar:
        loading_bar.set_description(notebook_path.name)
        clean_file(notebook_path.as_posix(), new=False)


if __name__ == "__main__":
    parser = ArgumentParser("Clean a notebook or notebooks in-place")
    parser.add_argument(
        "--notebook",
        type=str,
        required=True,
        help="Filename or regex to match notebooks against",
    )
    args = parser.parse_args()
    clean_notebook(filename=args.notebook)
