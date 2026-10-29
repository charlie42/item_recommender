import numpy as np
from sklearn.model_selection import StratifiedKFold
from item_recommender.hyperparameter_search import HyperparameterSearch
from item_recommender.evaluator import Evaluator
from item_recommender.feature_subset_extractor import FeatureSubsetExtractor
from item_recommender.model import make_model

class CrossValidator:
    """Perform cross-validation manually"""

    def __init__(self, split_data, output_column, config):
        self.config = config
        self.split_data = split_data
        self.output_column = output_column
        self.metric = config.get_item("metric")
        self.percentage_for_opt_n_features = config.get_item(
            "percentage for opt n features"
        )
        self.cv = StratifiedKFold(
            n_splits=config.get_item("outer cv n splits"),
            shuffle=True,
            random_state=0
        )

    def cross_validate(self):
        """Perform outer cross-validation on the training set."""
        X_train = self.split_data["X_train"]
        y_train = self.split_data["y_train"]

        print(f"Outer cross-validation for {self.output_column}")

        performances = []
        for i, (train_index, test_index) in enumerate(self.cv.split(X_train, y_train)):
            print(f"Fold {i + 1}/{self.cv.get_n_splits()}")

            _, performance = self._fit_and_evaluate(
                X_train.iloc[train_index],
                y_train.iloc[train_index],
                X_train.iloc[test_index],
                y_train.iloc[test_index]
            )
            performances.append(performance)

        return performances

    def recommend(self, cv_performances):
        """
        Select features on the whole training set and evaluate them on the test
        set, using the optimal number of features from cross-validation.
        """
        opt_n = self._get_opt_n(cv_performances)

        subsets, test_performances = self._fit_and_evaluate(
            self.split_data["X_train"],
            self.split_data["y_train"],
            self.split_data["X_test"],
            self.split_data["y_test"]
        )

        return {
            "items": subsets[opt_n],
            "test_performance": test_performances["perf_on_features"][opt_n],
            "test_performances": test_performances,
            "cv_performances": cv_performances,
        }

    def _get_opt_n(self, cv_performances):
        """
        Return the optimal number of features (first subset where mean
        performance reaches the specified percent of the max performance).
        """
        mean_perf = {
            n: np.mean([perf["perf_on_features"][n] for perf in cv_performances])
            for n in cv_performances[0]["perf_on_features"]
        }
        max_perf = max(mean_perf.values())
        for n in sorted(mean_perf):
            if mean_perf[n] >= max_perf * self.percentage_for_opt_n_features:
                return n

    def _fit_and_evaluate(self, X_train, y_train, X_test, y_test):
        search = HyperparameterSearch(X_train, y_train, self.config)
        search.search()
        params = search.best_params

        subsets = FeatureSubsetExtractor(params, self.config).get_subsets(
            X_train, y_train
        )

        performance = {
            "perf_all_features": self._evaluate_performance(
                params, X_train, y_train, X_test, y_test
            ),
            "perf_on_features": {
                n: self._evaluate_performance(
                    params, X_train[features], y_train, X_test[features], y_test
                )
                for n, features in subsets.items()
            },
        }
        return subsets, performance

    def _evaluate_performance(self, params, X_train, y_train, X_test, y_test):
        model = make_model(**params).fit(X_train, y_train)
        evaluator = Evaluator(model, X_test, y_test)
        return evaluator.evaluate(self.metric)
