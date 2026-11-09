import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, recall_score

class Evaluator:
    """Evaluate the performance of a model"""

    def __init__(self, model, X_test, y_test):
        self.model = model
        self.y_test = y_test
        self.y_pred = model.predict_proba(X_test)[:, 1]

    def evaluate(self):
        """Return the ROC AUC."""
        return roc_auc_score(self.y_test, self.y_pred)

    def get_threshold(self, target_sensitivity):
        """Return the highest probability threshold that reaches the target sensitivity."""
        _, tpr, thresholds = roc_curve(self.y_test, self.y_pred, drop_intermediate=False)
        return thresholds[np.argmax(tpr >= target_sensitivity)]

    def get_sensitivity_specificity(self, threshold):
        """Return sensitivity and specificity at the probability threshold."""
        y_pred = self.y_pred >= threshold
        return (
            recall_score(self.y_test, y_pred),
            recall_score(self.y_test, y_pred, pos_label=0),
        )
