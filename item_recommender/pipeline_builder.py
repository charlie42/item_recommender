from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import RFE
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier

class PipelineBuilder:
    def __init__(self, pipeline_config):
        self.pipeline_config = pipeline_config

    def get_pipeline(self):
        """Return the pipeline."""
        return self._build_pipeline()
    
    def _build_pipeline(self):
        """Build the pipeline."""
        steps = self.pipeline_config["steps"]
        parsed_steps = self._parse_steps(steps)
        return Pipeline(parsed_steps)

    def _parse_steps(self, steps):
        """
        Parse the steps from the config.
        E.g.: [imputer, scaler, fs1, fs2, model]

        Return a dictionary with the steps.
        """
        parsed_steps = []
        for step in steps:
            step_name = step["name"]

            config_reader = Config()

            parsed_steps.append(
                (step_name, config_reader.parse_item(step))
            )

        return parsed_steps