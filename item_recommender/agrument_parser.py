import argparse

class ArgumentParser:
    """Argument parser for the recommender script."""

    def __init__(self):
        pass

    def parse_arguments(self):
        """Parse command-line arguments."""
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--dataset-path", 
            type=str, 
            help="Path to the dataset file")
        parser.add_argument(
            "--config-path", 
            type=str, 
            help="Path to the config file")
        parser.add_argument(
            "--mode", 
            type=str, 
            help="Mode of the script: DEV, DEBUG, or PROD")
        return parser.parse_args()