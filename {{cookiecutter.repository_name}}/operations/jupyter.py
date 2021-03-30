import argparse
import subprocess
from pathlib import Path
from notebook.services.contents.filemanager import FileContentsManager


def jupyter(notebook_name, port):
    notebook_path = Path("notebooks") / notebook_name
    if not notebook_path.exists():
        FileContentsManager().new(path=notebook_path.as_posix())

    try:
        subprocess.run(["jupyter", "notebook", notebook_path.as_posix()])
    except KeyboardInterrupt:
        print()
        print("KeyboardInterrupt, exporting HTML before quitting")
    subprocess.run(["jupyter", "nbconvert", "--to", "html", notebook_path.as_posix()])
    html_path = notebook_path.with_suffix(".html")
    html_path.rename(html_path.name)  # move to current operation directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--notebook", type=str, required=True)
    parser.add_argument("--port", type=int, default=8888)
    args = parser.parse_args()
    jupyter(
        notebook_name=args.notebook,
        port=args.port,
    )
