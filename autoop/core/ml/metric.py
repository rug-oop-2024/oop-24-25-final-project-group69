from abc import ABC, abstractmethod
from typing import Any
import numpy as np

METRICS = [
    "mean_squared_error",
    "mean_absolute_error",
    "R_squared",
    "accuracy",
    "micro_precision",
    "micro_recall"
]

def get_metric(name: str) -> "Metric":
    """Factory function to get a metric by name.

    Args:
        name(str): the name of the metric as a string

    Returns:
        Metric: a metric instance given its str name
    """

    if name == "mean_squared_error":
        return MeanSquaredError()
    if name == "mean_absolute_error":
        return MeanAbsoluteError()
    if name == "R_squared":
        return RSquared()
    if name == "accuracy":
        return Accuracy()
    if name == "micro_precision":
        return MicroPrecision()
    if name == "micro_recall":
        return MicroRecall()


class Metric(ABC):
    
    """Base class for all metrics.

    """
    # metrics take ground truth and prediction as input and return a real number
    
    @abstractmethod
    def __call__(self, ground_truths: np.array, predictions: np.array) -> float:
        """
        Abstract implementation of the __call__ method for
        each metric to be calculated when the class is called
        
        Args:
            ground_truths(np.array): one-dimensional array with ground truths
            predictions(np.array): one-dimensional array with predictions
            
        Returns:
            float: the value of the metric
        """
        pass
    
    @abstractmethod
    def evaluate(self, predictions: np.array, ground_truths: np.array) -> float:
        """
        Abstract implementation of the __call__ method for
        each metric to be calculated when the class is called
        
        Args:
            ground_truths(np.array): one-dimensional array with ground truths
            predictions(np.array): one-dimensional array with predictions
            
        Returns:
            float: the value of the metric
        """
        pass

class MeanSquaredError(Metric):
    """Class for the mean squared error metric

    Inherits from:
        Metric
    """
    def __call__(self, ground_truths: np.array, predictions: np.array) -> float:
        """
        Implementation of the __call__ method for
        the MSE to be calculated when the class is called
        with the formula: 
        
        Args:
            ground_truths(np.array): one-dimensional array with ground truths
            predictions(np.array): one-dimensional array with predictions
        
        Returns:
            float: the value of MSE
        """
        return np.square(np.subtract(ground_truths, predictions)).mean()
    
    def evaluate(self, predictions: np.array, ground_truths: np.array) -> float:
        """
        Implementation of the __call__ method for
        the MSE to be calculated when the class is called
        with the formula: 
        
        Args:
            ground_truths(np.array): one-dimensional array with ground truths
            predictions(np.array): one-dimensional array with predictions
        
        Returns:
            float: the value of MSE
        """
        return np.square(np.subtract(ground_truths, predictions)).mean()

class MeanAbsoluteError(Metric):
    """Class for the MAE metric

    Inherits from:
        Metric
    """
    def __call__(self, ground_truths: np.array, predictions: np.array) -> float:
        """
        Implementation of the __call__ method for
        the MAE to be calculated when the class is called

        Args:
            ground_truths(np.array): one-dimensional array with ground truths
            predictions(np.array): one-dimensional array with predictions

        Returns:
            float: the value of MAE
        """
        return np.abs(np.subtract(ground_truths, predictions)).mean()
    
    def evaluate(self, predictions: np.array, ground_truths: np.array) -> float:
        """
        Implementation of the __call__ method for
        the MAE to be calculated when the class is called

        Args:
            ground_truths(np.array): one-dimensional array with ground truths
            predictions(np.array): one-dimensional array with predictions

        Returns:
            float: the value of MAE
        """
        return np.abs(np.subtract(ground_truths, predictions)).mean()
      
class RSquared(Metric):
    """Class for the R^2 metric

    Inherits from:
        Metric
    """
    def __call__(self, ground_truths: np.array, predictions: np.array) -> float:
        """
        Implementation of the __call__ method for
        the R^2 to be calculated when the class is called

        Args:
            ground_truths(np.array): one-dimensional array with ground truths
            predictions(np.array): one-dimensional array with predictions

        Returns:
            float: the value of R^2
        """
        corr_matrix = np.corrcoef(ground_truths, predictions)
        corr = corr_matrix[0,1]
        R_sq = corr**2
        
        return R_sq
    
    def evaluate(self, predictions: np.array, ground_truths: np.array) -> float:
        """
        Implementation of the __call__ method for
        the R^2 to be calculated when the class is called

        Args:
            ground_truths(np.array): one-dimensional array with ground truths
            predictions(np.array): one-dimensional array with predictions

        Returns:
            float: the value of R^2
        """
        corr_matrix = np.corrcoef(ground_truths, predictions)
        corr = corr_matrix[0,1]
        R_sq = corr**2
        
        return R_sq

class Accuracy(Metric):
    """Class for the accuracy metric

    Inherits from:
        Metric
    """ 
    def __call__(self, ground_truths: np.array, predictions: np.array) -> float:
        """
        Implementation of the __call__ method for
        the accuracy to be calculated when the class is called

        Args:
            ground_truths(np.array): one-dimensional array with ground truths
            predictions(np.array): one-dimensional array with predictions

        Returns:
            float: the value of accuracy
        """
        return np.mean(ground_truths == predictions)
    
    def evaluate(self, predictions: np.array, ground_truths: np.array) -> float:
        """
        Implementation of the __call__ method for
        the accuracy to be calculated when the class is called

        Args:
            ground_truths(np.array): one-dimensional array with ground truths
            predictions(np.array): one-dimensional array with predictions

        Returns:
            float: the value of accuracy
        """
        return np.mean(ground_truths == predictions)

class MicroPrecision(Metric):
    """Class for the micro-averaged precision metric

    Inherits from:
        Metric
    """ 
    def __call__(self, ground_truths: np.array, predictions: np.array) -> float:
        """
        Implementation of the __call__ method for
        the micro-precision to be calculated when the class is called

        Args:
            ground_truths(np.array): one-dimensional array with ground truths
            predictions(np.array): one-dimensional array with predictions

        Returns:
            float: the value of micro-averaged precision
        """
        TP = np.sum(ground_truths & predictions)
        FP = np.sum((1 - ground_truths) & predictions)
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        return precision
    
    def evaluate(self, predictions: np.array, ground_truths: np.array) -> float:
        """
        Implementation of the __call__ method for
        the micro-precision to be calculated when the class is called

        Args:
            ground_truths(np.array): one-dimensional array with ground truths
            predictions(np.array): one-dimensional array with predictions

        Returns:
            float: the value of micro-averaged precision
        """
        TP = np.sum(ground_truths & predictions)
        FP = np.sum((1 - ground_truths) & predictions)
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        return precision


class MicroRecall(Metric):
    """Class for the micro-averaged recall metric

    Inherits from:
        Metric
    """ 
    def __call__(self, ground_truths: np.array, predictions: np.array) -> float:
        """
        Implementation of the __call__ method for
        the micro-recall to be calculated when the class is called.

        Args:
            ground_truths(np.array): one-dimensional array with ground truths
            predictions(np.array): one-dimensional array with predictions

        Returns:
            float: the value of micro-averaged recall
        """
        TP = np.sum(ground_truths & predictions)
        FN = np.sum(ground_truths & (1 - predictions))

        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        return recall
    
    def evaluate(self, predictions: np.array, ground_truths: np.array) -> float:
        """
        Implementation of the __call__ method for
        the micro-recall to be calculated when the class is called.

        Args:
            ground_truths(np.array): one-dimensional array with ground truths
            predictions(np.array): one-dimensional array with predictions

        Returns:
            float: the value of micro-averaged recall
        """
        TP = np.sum(ground_truths & predictions)
        FN = np.sum(ground_truths & (1 - predictions))

        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        return recall
