"""Metric loggers: the console, or Weights & Biases (--use_wandb true)."""


class ConsoleLogger:
    def log(self, metrics):
        print(metrics)

    def close(self):
        pass


class WandbLogger:
    def __init__(self, run):
        self.run = run

    def log(self, metrics):
        self.run.log(metrics)

    def close(self):
        import wandb
        wandb.finish()
