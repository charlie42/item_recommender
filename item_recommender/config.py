import yaml
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from scipy.stats import loguniform

class Config:
    """Parse the config from file, or return a specific item."""

    CLASS_DICT = {
        "randomized_search": RandomizedSearchCV,
        "logistic_regression": LogisticRegression,
        "histgradientboosting": HistGradientBoostingClassifier,
        "simple_imputer": SimpleImputer,
        "knn_imputer": KNNImputer,
        "standard_scaler": StandardScaler,
        "minmax_scaler": MinMaxScaler,
        "rfe": RFE,
        "sfs": SFS,
        "loguniform": loguniform,
        "stratified_kfold": StratifiedKFold,
        "pipeline": Pipeline,
    }

    def __init__(self):
        self.config = None
        
    def load_config(self, path, mode="DEBUG"):
        """Load the config from file, append mode."""
        config = yaml.safe_load(open(path, "r", encoding="utf-8"))
        config["mode"] = mode
        self.config = config
        return config
        
    def get_item(self, item_path):
        """
        Return a specific item from the config.
        Either: get_specific_item("outer_cv.n_splits")
        Or: get_specific_item("n_splits")
        If using the second option, returning the first item found.
        If item has a "class" key, return an instance of the class with the
        given parameters.
        """
        item_path = item_path.split(".")
        item = self.config
        for path in item_path:
            item = item[path]
        if "class" in item:
            return self._parse_class(item)
        return item
    
    def _parse_class(self, item):
        """
        Parse an class from the config.
        If the item has a "class" key, return an instance of the class with the
        given parameters.
        If class is "Pipeline", build a pipeline with the given steps.
        """
        if "class" in item:
            class_name = item["class"]
            class_params = item["params"]

            if class_name == "pipeline":
                from pipeline_builder import PipelineBuilder
                return PipelineBuilder().build_pipeline(class_params)
            else:
                return self._get_class(class_name, class_params)
        else:
            raise ValueError("Item does not have a class key.")
    
    def _get_class(self, class_name, class_params):
        """Return the class with the given name and parameters."""
        class_ = self.CLASS_DICT[class_name]
        return class_(**class_params)
    
    
        