import numpy as np 

def accuracy(yhat : np.ndarray, y: np.ndarray): 
    """
        yhat: (1, m) predicted class indices
        y:    (1, m) true class indices
        Return:
            scalar accuracy between 0 and 1
    """
    return float(np.mean(yhat == y)) 
def confusion_matrix(yhat: np.ndarray, y: np.ndarray, n_classes: int = 10):
    """
        yhat: (1, m) predicted class indices
        y:    (1, m) true class indices
        Return:
            confusion_matrix: (n_classes, n_classes)
            Diagonal     -- correct predictions
            Off-diagonal -- mistakes
    """
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y.flatten(), yhat.flatten()):
        cm[true][pred] += 1
    return cm