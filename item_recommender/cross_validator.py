from item_recommender.data_splitter import DataSplitter
from item_recommender.hyperparameter_search import HyperparameterSearch
from item_recommender.evaluator import Evaluator
from item_recommender.config import Config
from item_recommender.feature_subset_extractor import FeatureSubsetExtractor

class CrossValidator:
    """Perform cross-validation manually"""

    def __init__(self, split_data, output_column, config):
        self.config = config
        self.split_data = split_data
        self.output_column = output_column
        self.n_features_to_check = config["n_features_to_check"]
        self.metric = config["metric"]
        self.n_splits = config["outer_cv"]["n_splits"]
        self.n_features_to_evaluate = config["n_features_to_evaluate"]
        self.cv = Config().get_item(config["cv"])
        
        self.cv_search_objects = []
        self.performance = None

    def cross_validate(self):
        """Perform outer cross-validation on the training set."""
        X_train, _, y_train, _ = self.split_data

        for fold in self.cv.split(X_train, y_train):
            # Split the training set into train and test folds
            train_index, test_index = fold
            X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
            y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

            # Fit hyperparameter search on the training fold
            search_object = HyperparameterSearch(X_train_fold, y_train_fold, self.config)
            search_object.search()

            # Save the search object
            self.cv_search_objects.append(search_object)

            # Get the performance of the best model on the test fold
            self.performance = self._evaluate_performance_for_fold(
                search_object.best_model,
                X_test_fold,
                y_test_fold
            )

        return self.output_column, self.cv_search_objects, self.performance

    def _split_data(self, config):
        data_splitter = DataSplitter(self.data, self.output_column)
        return data_splitter.split_data(config)
    
    def _evaluate_performance_for_fold(self, model, X_test, y_test):
        # Evaluate performance on all features
        perf_all_features = self._evaluate_performance(model, X_test, y_test)

        # Evaluate performance on all checked features
        features = FeatureSubsetExtractor(model, self.config).get_n_feature_indices(
            self.n_features_to_check
        )
        perf_all_checked_features = self._evaluate_performance(
            model, 
            X_test.iloc[:, features], 
            y_test
        )

        # Evaluate performance on every subset
        opt_ns_of_features, perf_on_features = self._evaluate_performance_on_features(
            model, 
            X_test, 
            y_test
        )

        return {
            "perf_all_features": perf_all_features,
            "perf_all_checked_features": perf_all_checked_features,
            "opt_ns_of_features": opt_ns_of_features,
            "perf_on_features": perf_on_features
        }

    def _evaluate_performance_on_features(self, model, X_test, y_test):
        # Evaluate performance on all checked features
        perf_on_features = {}

        feature_selector = FeatureSubsetExtractor(model, self.config)

        for n_features in range(1, self.n_features_to_evaluate + 1):

            features = feature_selector.get_n_feature_indices(n_features)
            perf_on_features[n_features].append(self._evaluate_performance(
                model, 
                X_test.iloc[:, features], 
                y_test
            ))
    
        opt_n_of_features = feature_selector.get_opt_n()

        return opt_n_of_features, perf_on_features

    def _evaluate_performance(self, model, X_test, y_test):
        evaluator = Evaluator(model, X_test, y_test)
        return evaluator.evaluate(self.metric)