from sklearn.model_selection import StratifiedKFold
from item_recommender.data_splitter import DataSplitter
from item_recommender.hyperparameter_search import HyperparameterSearch

class CrossValidator:
    def __init__(self, data, output_column, config):
        self.data = data
        self.output_column = output_column
        self.n_features_to_check = config["n_features_to_check"]
        self.cv_search_objects = []
        self.performance = {
            "auc_all_features": [],
            "auc_all_checked_features": [],
            "opt_ns_of_features": [],
            "perf_on_features": {}
        }
        pass

    def split_data(self, config):
        data_splitter = DataSplitter(self.data, self.output_column)
        return data_splitter.split_data(config)

    def cross_validate(self, config):
        """Perform outer cross-validation on the training set."""
        n_splits = config["outer_cv"]["n_splits"]

        cv = StratifiedKFold(n_splits, shuffle=True, random_state=0)

        X_train, X_test, y_train, y_test = self.split_data(config)

        

        for fold in cv.split(X_train, y_train):
            train_index, test_index = fold
            X_train_fold, X_test_fold = X_train.iloc[train_index], X_train.iloc[test_index]
            y_train_fold, y_test_fold = y_train.iloc[train_index], y_train.iloc[test_index]

            # Fit hyperparameter search on the training fold
            search_object = HyperparameterSearch(X_train_fold, y_train_fold, config)
            search_object.search()

