from sklearn.pipeline import Pipeline
from item_recommender.config import Config

class PipelineBuilder:
    """Build a pipeline from the config."""

    def __init__(self):
        pass

    def build_pipeline(self, pipeline_config):
        """Build the pipeline."""
        steps = pipeline_config["steps"]
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
                (step_name, config_reader.get_item(step))
            )

        return parsed_steps