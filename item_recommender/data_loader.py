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
        return self.data

    def get_output_columns(self):
        """Return the output columns."""
        prefix = self.config["output column prefix"]
        print("DEBUG: prefix", prefix)
        print("DEBUG: self.data.columns", self.data.columns)
        print("DEBUG: ", [col for col in self.data.columns if col.startswith(prefix)])
        print("DEBUG: ", self.data.columns[-1].startswith(prefix))
        print("DEBUG: ", self.data.columns[-1])
        return [col for col in self.data.columns if col.startswith(prefix)]