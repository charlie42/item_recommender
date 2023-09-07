import yaml
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import RFE
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from scipy.stats import loguniform
from item_recommender.pipeline_builder import PipelineBuilder
from item_recommender.config_reader import Config
from sklearn.pipeline import Pipeline

class Config():

    _class_dict = {
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

    def __init__(self, path, mode):
        self.config = self.load_config(path)
        self.config["mode"] = mode # Append mode from args to config

    def _load_config(self, path):
        """Load the config from file."""
        return yaml.safe_load(open(path, "r"))
    
    def parse_item(self, item):
        """
        Parse an item from the config.
        If the item has a "class" key, return an instance of the class with the
        given parameters.
        If class is "Pipeline", build a pipeline with the given steps.
        """
        if "class" in item:
            class_name = item["class"]
            class_params = item["params"]

            if class_name == "pipeline":
                return PipelineBuilder(class_params).build_pipeline()
            else:
                return self._get_class(class_name, class_params)
        else:
            return item
        
    # def get_single_value(self, key_hierarchy):
    #     """Return the value for the given key hierarchy, 
    #     e.g. ["hp_search", "n_iter"]."""
    #     value = self.config
    #     for key in key_hierarchy:
    #         value = value[key]
    #     return value