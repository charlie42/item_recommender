from item_recommender.config import Config

class HyperparameterSearch:
    """Perform hyperparameter search."""

    def __init__(self, X_train, y_train, config):
        self.X_train = X_train
        self.y_train = y_train
        self.config = config
        self.search_instance = self._get_search_object()

    def _get_search_object(self):
        """Return the search object with params from config."""
        return self.config.get_item("outer cv.hp search")
    
    def search(self):
        """Perform hyperparameter search."""
        print("DEBUG: self.search_instance", self.search_instance)
        [print("DEBUG: ", param) for param in self.search_instance.get_params().keys()]
        self.search_instance.fit(self.X_train, self.y_train)

    @property
    def best_model(self):
        """Return the best model from the search."""
        return self.search_instance.best_estimator_