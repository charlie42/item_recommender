from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedKFold
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from item_recommender.model import make_model

class FeatureSubsetExtractor:
    """Get the best feature subset of each size with RFE then SFS"""

    def __init__(self, model_params, config):
        self.model_params = model_params
        self.n_features_to_check = config.get_item("n features to check")
        self.metric = config.get_item("metric")
        self.n_splits = config.get_item("inner cv n splits")

    def get_subsets(self, X, y):
        """Return a dictionary of {n: names of the best n features}."""
        rfe = RFE(
            make_model(**self.model_params),
            n_features_to_select=self.n_features_to_check,
            importance_getter="named_steps.model.coef_",
        )
        rfe.fit(X, y)
        rfe_columns = X.columns[rfe.support_]

        sfs = SFS(
            make_model(**self.model_params),
            k_features=self.n_features_to_check,
            forward=True,
            floating=True,
            scoring=self.metric,
            cv=StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=0),
        )
        sfs.fit(X[rfe_columns], y)

        return {
            n: list(rfe_columns[list(subset["feature_idx"])])
            for n, subset in sfs.get_metric_dict().items()
        }
