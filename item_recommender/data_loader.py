import pandas as pd

class DataLoader:
    """Loads data from a file."""

    def __init__(self, path, config):
        self.path = path
        self.config = config
        self.data = None

    def load_data(self):
        """Load the data from the file."""
        self.data = pd.read_csv(self.path)

    def get_output_columns(self):
        """Return the output columns."""
        prefix = self.config["output column prefix"]
        return [col for col in self.data.columns if col.startswith(prefix)]