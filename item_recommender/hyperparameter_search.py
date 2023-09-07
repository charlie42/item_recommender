class HyperparameterSearch:
    def __init__(self, X_train, y_train, config):
        self.X_train = X_train
        self.y_train = y_train
        self.config = config

    def _get_search_object(self):
        """Return the search object with params from config."""
        config_reader = Config()
        return config_reader.parse_item(self.config["hp_search"])
    
    def search(self):
        """Perform hyperparameter search."""
    
        search_object = self._get_search_object()
        search_object.fit(self.X_train, self.y_train)