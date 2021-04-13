from guild.commands.run import run


if __name__ == "__main__":
    run(
        [
            "evaluate",
            "-y",
            "n_workers=0",
            "eval_batch_size=4",
            "--debug-sourcecode=.",
            "--label debug",
        ]
    )
