import argparse


class Options(object):
    def __init__(self):

        # Handle command line arguments
        self.parser = argparse.ArgumentParser(
            description="Run a complete training pipeline. Optionally, a JSON configuration file can be used, to overwrite command-line arguments."
        )

        # Run from config file
        self.parser.add_argument(
            "--config",
            dest="config_filepath",
            help="Configuration .json file (optional). Overwrites existing command-line args!",
        )

        # Run from command-line arguments
        # I/O
        self.parser.add_argument(
            "--output_dir",
            default="./output",
            help="Root output directory. Must exist. Time-stamped directories will be created inside.",
        )
        self.parser.add_argument("--data_dir", default="./data", help="Data directory")
        self.parser.add_argument("--load_model", help="Path to pre-trained model.")
        self.parser.add_argument(
            "--name",
            dest="experiment_name",
            default="",
            help="A string identifier/name for the experiment to be run - it will be appended to the output directory name, before the timestamp",
        )
        self.parser.add_argument(
            "--max_seq_len",
            type=int,
            help="""Maximum input sequence length. Determines size of transformer layers.
                                 If not provided, then the value defined inside the data class will be used.""",
        )
        self.parser.add_argument(
            "--data_window_len",
            type=int,
            help="""Used instead of the `max_seq_len`, when the data samples must be
                                 segmented into windows. Determines maximum input sequence length 
                                 (size of transformer layers).""",
        )
        self.parser.add_argument(
            "--test",
            action="store_true",
            help="only test model, no training",
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

    def parse(self):

        args = self.parser.parse_args()
        return args
