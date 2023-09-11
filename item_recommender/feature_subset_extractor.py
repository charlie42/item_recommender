class FeatureSubsetExtractor:
    """Get feature subsets from a fitted pipeline"""

    FEATURE_SELECTORS = ["rfe", "sfs"]
    
    def __init__(self, pipeline, config):
        self.fitted_feature_selector = self._get_feature_selector(pipeline)
        self.percentage_for_opt_n_features = config.get_specific_item(
            "percentage_for_opt_n_features"
        )

    def _get_feature_selector(self, pipeline):
        """
        Return the last feature selector from the pipeline from the 
        hyperparameter search object. 
        """
        step_names = [step[0] for step in pipeline.steps]
        for name in reversed(step_names):
            if name in self.FEATURE_SELECTORS:
                return pipeline.named_steps[name]
            
    def get_n_feature_indices(self, n):
        """Return the indices of the n most important features."""
        return self.fitted_feature_selector.get_metric_dict()[n]["feature_idx"]
    
    def get_opt_n(self):
        """
        Return the optimal number of features (first subset where performance 
        reaches the specified percent of the max oerformance).
        """

        # Get the max performance from get_metric_dict()[SUBSET]['avg_score']
        metric_dict = self.fitted_feature_selector.get_metric_dict()
        max_perf = max([metric_dict[subset]["avg_score"] for subset in metric_dict])

        # Get the optimal number of features
        for subset in metric_dict:
            # If the performance is within the specified percent of the max performance
            if metric_dict[subset]["avg_score"] >= max_perf * self.percentage_for_opt_n_features:
                return subset
        
        