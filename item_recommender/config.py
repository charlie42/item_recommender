import yaml

class Config:
    """Parse the config from file, or return a specific item."""

    def __init__(self):
        self.config = None

    def load_config(self, path, mode):
        """Load the config from file, apply the overrides for the mode."""
        with open(path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        config_dict.update(config_dict.pop("modes")[mode])
        self.config = config_dict

    def get_item(self, item_path):
        """Return a specific item from the config."""
        return self.config[item_path]
