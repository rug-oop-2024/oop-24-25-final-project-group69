from abc import ABC, abstractmethod
import numpy as np

METRICS = [
    "Mean Squared Error",
    "Mean Absolute Error",
    "R Squared",
    "Accuracy",
    "Micro Precision",
    "Micro Recall"
]


def get_metric(name: str) -> "Metric":
    """Factory function to get a metric by name.

    Args:
        name (str): the name of the metric as a string.

    Returns:
        Metric: a metric instance corresponding to the given name.
    """
    if name == "Mean Squared Error":
        return MeanSquaredError()
    if name == "Mean Absolute Error":
        return MeanAbsoluteError()
    if name == "R Squared":
        return RSquared()
    if name == "Accuracy":
        return Accuracy()
    if name == "Micro Precision":
        return MicroPrecision()
    if name == "Micro Recall":
        return MicroRecall()


class Metric(ABC):
    """Base class for all metrics.

    Metrics take ground truth and prediction as input and return a real
    number.
    """

    @abstractmethod
    def __str__(self) -> str:
        """Return the string description of the metric."""
        pass

    @abstractmethod
    def evaluate(self, predictions: np.ndarray,
                 ground_truths: np.ndarray) -> float:
        """Evaluate the metric based on
        the given predictions and ground truths.

        Args:
            predictions (np.array): One-dimensional array of predicted values.
            ground_truths (np.array): One-dimensional array of true values.

        Returns:
            float: The evaluated metric value.
        """
        pass


class MeanSquaredError(Metric):
    """Class for the Mean Squared Error (MSE) metric."""

    def __str__(self) -> str:
        """Return a string description of the
        Mean Squared Error metric."""
        return "Mean Squared Error Metric"

    def evaluate(self, predictions: np.ndarray,
                 ground_truths: np.ndarray) -> float:
        """
        Evaluate the Mean Squared Error (MSE) based on predictions and ground
        truths.

        In this case, the ground truths and predictions are passed as they are.
        There's no transformation (such as binarization or reshaping) required
        for this metric.

        Args:
            predictions (np.ndarray): 1D array of predicted values.
            ground_truths (np.ndarray): 1D array of true values (labels).

        Returns:
            float: The computed MSE value.
        """
        return np.square(np.subtract(ground_truths, predictions)).mean()


class MeanAbsoluteError(Metric):
    """Class for the Mean Absolute Error (MAE) metric."""

    def __str__(self) -> str:
        """Return a string description of the Mean Absolute Error metric."""
        return "Mean Absolute Error Metric"

    def evaluate(self, predictions: np.ndarray,
                 ground_truths: np.ndarray) -> float:
        """
        Evaluate the Mean Absolute Error (MAE) based on predictions and ground
        truths.

        No transformations are needed for the evaluation of MAE. Both the
        ground truths and predictions are used directly in their current form.

        Args:
            predictions (np.ndarray): 1D array of predicted values.
            ground_truths (np.ndarray): 1D array of true values (labels).

        Returns:
            float: The computed MAE value.
        """
        return np.abs(np.subtract(ground_truths, predictions)).mean()


class RSquared(Metric):
    """Class for the R^2 metric (Coefficient of Determination)."""

    def __str__(self) -> str:
        """Return a string description of the R^2 metric."""
        return "R Squared Metric"

    def evaluate(self, predictions: np.ndarray,
                 ground_truths: np.ndarray) -> float:
        """
        Evaluate the R^2 score based on predictions and ground truths.

        In the case of R^2, the `ground_truths` and `predictions` are both
        flattened (i.e., converted to one-dimensional arrays) to ensure they
        are correctly aligned. The correlation coefficient between the two is
        calculated, and the R^2 value is returned.

        Args:
            predictions (np.ndarray): 1D array of predicted values.
            ground_truths (np.ndarray): 1D array of true values (labels).

        Returns:
            float: The computed R^2 score.
        """
        ground_truths = ground_truths.flatten()
        predictions = predictions.flatten()

        # Check for zero variance to avoid NaN
        if np.std(ground_truths) == 0 or np.std(predictions) == 0:
            return 0.0

        corr_matrix = np.corrcoef(ground_truths, predictions)
        corr = corr_matrix[0, 1]

        # Avoid NaN again
        if np.isnan(corr):
            return 0.0

        R_sq = corr ** 2
        return R_sq


class Accuracy(Metric):
    """Class for the Accuracy metric."""

    def __str__(self) -> str:
        """Return a string description of the Accuracy metric."""
        return "Accuracy Metric"

    def evaluate(self, predictions: np.ndarray,
                 ground_truths: np.ndarray) -> float:
        """
        Evaluate the Accuracy based on predictions and ground truths.

        This metric does not require any transformation of the input arrays.
        The `ground_truths` and `predictions` are directly compared to
        calculate the accuracy (i.e., the percentage of correct predictions).

        Args:
            predictions (np.ndarray): 1D array of predicted values.
            ground_truths (np.ndarray): 1D array of true values (labels).

        Returns:
            float: The computed accuracy.
        """
        return np.mean(ground_truths == predictions)


class MicroPrecision(Metric):
    """Class for the micro-averaged Precision metric."""

    def __str__(self) -> str:
        """Return a string description of the Micro Precision metric."""
        return "Micro Precision Metric"

    def evaluate(self, predictions: np.ndarray,
                 ground_truths: np.ndarray) -> float:
        """
        Evaluate the Micro Precision based on predictions and ground truths.

        For micro-precision, the `ground_truths` and `predictions` are
        transformed into boolean arrays, where 1 indicates a positive class
        and 0 indicates a negative class. Then, the true positives (TP) and
        false positives (FP) are computed, and the micro-averaged precision is
        calculated.

        Args:
            predictions (np.ndarray): 1D array of predicted values (0 or 1).
            ground_truths (np.ndarray): 1D array of true values (0 or 1).

        Returns:
            float: The computed micro-averaged precision.
        """
        ground_truths = ground_truths.astype(bool)
        predictions = predictions.astype(bool)

        TP = np.sum(ground_truths & predictions)
        FP = np.sum((1 - ground_truths) & predictions)
        return TP / (TP + FP) if (TP + FP) > 0 else 0


class MicroRecall(Metric):
    """Class for the micro-averaged Recall metric."""

    def __str__(self) -> str:
        """Return a string description of the Micro Recall metric."""
        return "Micro Recall Metric"

    def evaluate(self, predictions: np.ndarray,
                 ground_truths: np.ndarray) -> float:
        """
        Evaluate the Micro Recall based on
        predictions and ground truths.

        Similar to micro-precision, the
        `ground_truths` and `predictions` are
        transformed into boolean arrays.
        The true positives (TP) and false
        negatives (FN) are computed, and
        the micro-averaged recall is calculated.

        Args:
            predictions (np.ndarray): 1D array
            of predicted values (0 or 1).
            ground_truths (np.ndarray): 1D array of true values (0 or 1).

        Returns:
            float: The computed micro-averaged recall.
        """
        ground_truths = ground_truths.astype(bool)
        predictions = predictions.astype(bool)

        TP = np.sum(ground_truths & predictions)
        FN = np.sum(ground_truths & (1 - predictions))
        return TP / (TP + FN) if (TP + FN) > 0 else 0
