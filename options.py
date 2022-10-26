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

    def parse(self):
        args = self.parser.parse_args()
        return args
