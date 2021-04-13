from guild.commands.run import run


if __name__ == "__main__":
    run(
        [
            "train",
            "-y",
            "n_workers=0",
            "n_batches_per_epoch=2",
            "batch_size=4",
            "eval_batch_size=4",
            "--debug-sourcecode=.",
            "--label debug",
        ]
    )
