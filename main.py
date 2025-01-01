from lightning.pytorch.cli import LightningCLI

from data import LanguageModelDataModule
from model import GPTLightning, GPTCallback


def cli_main():
    cli = LightningCLI(GPTLightning, LanguageModelDataModule, trainer_defaults={
            "max_epochs": 200,
            "callbacks": [
                GPTCallback(log_every_n_steps=10, accumulate_grad_batches=1)
            ]
        })


if __name__ == "__main__":
    cli_main()


