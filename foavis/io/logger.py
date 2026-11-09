"""Metric loggers: the console, or Weights & Biases (--use_wandb true)."""


class ConsoleLogger:
    def __init__(self, enabled=True):   # under torchrun only the main process prints
        self.enabled = enabled

    def log(self, metrics):
        if self.enabled:
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
