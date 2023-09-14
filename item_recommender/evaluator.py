from sklearn.metrics import roc_auc_score
from scipy.stats import gmean

class Evaluator:
    """Evaluate the performance of a model"""

    METRIC_DICT = {
        "roc_auc": roc_auc_score,
        "gmean": gmean,
    }

    def __init__(self, model, X_test, y_test):
        self.model = model
        self.X_test = X_test
        self.y_test = y_test

    def evaluate(self, metric):
        """Evaluate the model."""
        if metric in self.METRIC_DICT.keys():
            return self._evaluate_metric(metric, predict_proba=True)
        else:
            raise ValueError(f"Metric {metric} not supported.")
        
    def _evaluate_metric(self, metric, predict_proba=False):
        """Evaluate the model using the given metric."""
        y_pred = self._predict_data(predict_proba)
        return self.METRIC_DICT[metric](self.y_test, y_pred)
    
    def _predict_data(self, predict_proba=False):
        """Predict the data."""
        if predict_proba:
            return self.model.predict_proba(self.X_test)[:, 1]
        else:
            return self.model.predict(self.X_test)