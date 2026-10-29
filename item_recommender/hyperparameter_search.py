from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from scipy.stats import loguniform
from item_recommender.model import make_model

class HyperparameterSearch:
    """Perform hyperparameter search."""

    def __init__(self, X_train, y_train, config):
        self.X_train = X_train
        self.y_train = y_train
        self.config = config
        self.search_instance = self._get_search_object()

    def _get_search_object(self):
        """Return the search object with params from config."""
        return RandomizedSearchCV(
            make_model(),
            param_distributions={
                "model__C": loguniform(0.001, 1000),
                "model__class_weight": ["balanced", None],
            },
            n_iter=self.config.get_item("hp search n iter"),
            scoring=self.config.get_item("metric"),
            cv=StratifiedKFold(
                n_splits=self.config.get_item("inner cv n splits"),
                shuffle=True,
                random_state=0
            ),
            error_score="raise",
            random_state=0,
        )

    def search(self):
        """Perform hyperparameter search."""
        self.search_instance.fit(self.X_train, self.y_train)

    @property
    def best_params(self):
        """Return the best model params from the search."""
        return {
            name.removeprefix("model__"): value
            for name, value in self.search_instance.best_params_.items()
        }
