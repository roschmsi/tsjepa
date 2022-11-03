import argparse


class Options(object):
    def __init__(self):

        # Handle command line arguments
        self.parser = argparse.ArgumentParser(description="Run pipeline for all models")

        # Run from config file
        self.parser.add_argument(
            "--config",
            dest="config_path",
            help="Configuration .json file (optional). Overwrites existing command-line args!",
        )
        self.parser.add_argument("--load_model", help="Path to pre-trained model.")
        self.parser.add_argument(
            "--test",
            action="store_true",
            help="only test model, no training",
        )
        self.parser.add_argument(
            "--debug",
            action="store_true",
            help="debug model with single sample",
        )
        self.parser.add_argument(
            "--print_interval",
            type=int,
            default=100,
            help="Print batch info every this many batches",
        )
        self.parser.add_argument(
            "--console",
            action="store_true",
            help="Optimize printout for console output; otherwise for file",
        )
        self.parser.add_argument(
            "--key_metric",
            choices={"loss", "accuracy", "precision"},
            default="loss",
            help="Metric used for defining best epoch",
        )
        self.parser.add_argument(
            "--save_all",
            action="store_true",
            help="If set, will save model weights (and optimizer state) for every epoch; otherwise just latest",
        )
        self.parser.add_argument(
            "--change_output",
            action="store_true",
            help="change unsupervised model to supervised model",
        )
        self.parser.add_argument(
            "--val_interval",
            type=int,
            default=2,
            help="Evaluate on validation set every this many epochs. Must be >= 1.",
        )
        self.parser.add_argument(
            "--resume",
            action="store_true",
            help="If set, will load `starting_epoch` and state of optimizer, besides model weights.",
        )

    def parse(self):
        args = self.parser.parse_args()
        return args
