import yaml
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.pipeline import Pipeline
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from scipy.stats import loguniform, uniform
from item_recommender.mode_manager import ModeManager

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
        "uniform": uniform,
        "stratified_k_fold": StratifiedKFold,
        "pipeline": Pipeline,
    }

    def __init__(self):
        self.config = None
        
    def load_config(self, path, mode):
        """Load the config from file, append mode."""
        config_dict = yaml.safe_load(open(path, "r", encoding="utf-8"))
        config_dict = ModeManager(config_dict, mode).update_config()
        config_dict = self.parse_classes(config_dict)
        self.config = config_dict
        
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
            print("DEBUG: path", path)
            item = item[path]
        if isinstance(item, dict) and "class" in item:
            print("DEBUG: has class")
            return self._parse_class(item)
        return item
    
    def set_item(self, item_path, value):
        """
        Set a specific item in the config.
        Either: set_specific_item("outer_cv.n_splits", 10)
        Or: set_specific_item("n_splits", 10)
        If using the second option, setting the first item found.
        """
        item_path = item_path.split(".")
        item = self.config
        for path in item_path[:-1]:
            item = item[path]
        item[item_path[-1]] = value

    def parse_classes(self, config_dict):
        """
        Go through each item, and replace each item that has "class" key with an
        instance of the class with the given parameters. 
        """
        # If "class" anywhere in the lower levels, parse that class recursively
        for key, item in config_dict.items():
            if isinstance(item, dict):
                config_dict[key] = self.parse_classes(item)

        # If "class" in the current level, parse that class
        if "class" in config_dict:
            config_dict = self._parse_class(config_dict)

        return config_dict
            
    
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

            print("DEBUG: class_name", class_name, "class_params", class_params)

            if class_name == "pipeline":
                from item_recommender.pipeline_builder import PipelineBuilder
                print("DEBUG: pipeline")
                print("DEBUG: PipelineBuilder().build_pipeline(class_params)", PipelineBuilder().build_pipeline(class_params))
                return PipelineBuilder().build_pipeline(class_params)
            else:
                return self._get_class(class_name, class_params)
        else:
            raise ValueError("Item does not have a class key.")
    
    def _get_class(self, class_name, class_params):
        """Return the class with the given name and parameters."""
        class_ = self.CLASS_DICT[class_name]
        print("DEBUG: class_", class_)
        if isinstance(class_params, dict):
            return class_(**class_params)
        elif isinstance(class_params, list):
            return class_(*class_params)
    
    
        