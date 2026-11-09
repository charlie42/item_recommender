import pandas as pd

class DataLoader:
    """Loads data from a file."""

    def __init__(self, path, config):
        self.path = path
        self.config = config
        self.data = None

    def load_data(self):
        """Load the data, drop excluded columns and items with too many missing values."""
        self.data = pd.read_csv(self.path, index_col=None)
        self.data = self.data.drop(columns=self.config.get_item("exclude columns"))
        items = self.data.drop(columns=self.get_output_columns())
        too_sparse = items.columns[
            items.isna().mean() > self.config.get_item("max missing fraction")
        ]
        print("Dropping items with too many missing values:", list(too_sparse))
        self.data = self.data.drop(columns=too_sparse)
        return self.data

    def get_output_columns(self):
        """Return the output columns."""
        prefix = self.config.get_item("output column prefix")
        return [col for col in self.data.columns if col.startswith(prefix)]

    def get_output_columns_to_process(self, output_columns):
        """Return the output columns where both classes have enough samples."""
        min_class_size = self.config.get_item("min class size")
        return [
            col for col in output_columns
            if self.data[col].value_counts().reindex([0, 1], fill_value=0).min()
            >= min_class_size
        ]