import torch
import numpy as np
from sklearn.metrics import top_k_accuracy_score, classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def accuracy(target: np.array, output: np.array):
    """
    Accuracy: TP + TN / (TP + TN + FP + FN)
    equivalent to:
        pred = np.argmax(output, axis=1)
        correct = np.sum(pred == target)
        return correct / len(target)
    """
    pred = np.argmax(output, axis=1)
    assert pred.shape[0] == len(target)
    return accuracy_score(target, pred)


def confusion_matrix(target: np.array, output: np.array, num_classes: int):
    """
    Accumulates the confusion matrix
    """
    conf_matrix = np.zeros([num_classes, num_classes])
    pred = np.argmax(output, 1)
    for t, p in zip(target.reshape(-1), pred.reshape(-1)):
        conf_matrix[t, p] += 1
    return conf_matrix


def precision(target: np.array, output: np.array, average: str = "weighted"):
    """
    Precision: TP / (TP + FP)
    """
    valid_avgs = {"micro", "macro", "weighted"}
    if average not in valid_avgs:
        raise ValueError(f"{average} mode is not supported. Use from {valid_avgs}")
    pred = np.argmax(output, axis=1)
    assert pred.shape[0] == len(target)
    return precision_score(target, pred, average=average)


def recall(target: np.array, output: np.array, average: str = "weighted"):
    """
    Recall: TP / (TP + FN)
    """
    valid_avgs = {"micro", "macro", "weighted"}
    if average not in valid_avgs:
        raise ValueError(f"{average} mode is not supported. Use from {valid_avgs}")
    pred = np.argmax(output, axis=1)
    assert pred.shape[0] == len(target)
    return recall_score(target, pred, average=average)


def f1score(target: np.array, output: np.array, average: str = "weighted"):
    """
    F1 score: (2 * p * r) / (p + r)
    """
    valid_avgs = {"micro", "macro", "weighted"}
    if average not in valid_avgs:
        raise ValueError(f"{average} mode is not supported. Use from {valid_avgs}")
    pred = np.argmax(output, axis=1)
    assert pred.shape[0] == len(target)
    return f1_score(target, pred, average=average)


def accuracy_mse(target: np.array, output: np.array):
    """
    Accuracy when using regression rather than classification.
    """
    assert len(output) == len(target)
    correct = np.sum(((output - target).abs() < 1))
    return correct / len(target)


def acc_per_class(target: np.array, output: np.array, num_classes: int):
    """
    Calculates acc per class
    """
    conf_mat = confusion_matrix(target, output, num_classes)
    return conf_mat.diagonal() / conf_mat.sum(1)


def top_k_acc(target: np.array, output: np.array, k: int):
    return top_k_accuracy_score(y_true=target, y_score=output, k=k)


def top_1_acc(target: np.array, output: np.array):
    return top_k_acc(target, output, k=1)


def top_2_acc(target: np.array, output: np.array):
    return top_k_acc(target, output, k=2)


def top_3_acc(target: np.array, output: np.array):
    return top_k_acc(target, output, k=3)


def top_4_acc(target: np.array, output: np.array):
    return top_k_acc(target, output, k=4)


def top_5_acc(target: np.array, output: np.array):
    return top_k_acc(target, output, k=5)


def classification_report_sklearn(target: np.array, output: np.array, target_names: list = None):
    pred = np.argmax(output, axis=1)
    assert pred.shape[0] == len(target)
    return classification_report(target, pred, target_names=target_names)


def accuracy_torch(output: torch.tensor, target: torch.tensor):
    """
    Accuracy: TP + TN / (TP + TN + FP + FN)
    """
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def accuracy_mse_torch(target: torch.tensor, output: torch.tensor):
    """
    Accuracy when using regression rather than classification.
    """
    with torch.no_grad():
        assert len(output) == len(target)
        correct = 0
        correct += torch.sum(((output - target).abs() < 1)).item()
        return correct / len(target)


def top_k_acc_torch(target: torch.tensor, output: torch.tensor, k: int):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def acc_relaxed_torch(target: torch.tensor, output: torch.tensor):
    """
    Function so that 101 age classes correspond to 8 age classes,
    for Adience dataset.
    This results in the same value as the vanilla accuracy.
    """
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)

        correct = 0
        for p, t in zip(pred, target):
            if (0 <= p < 3) and (0 <= t < 3):
                correct += 1
            elif (3 <= p < 7) and (3 <= t < 7):
                correct += 1
            elif (7 <= p < 13.5) and (7 <= t < 13.5):
                correct += 1
            elif (13.5 <= p < 22.5) and (13.5 <= t < 22.5):
                correct += 1
            elif (22.5 <= p < 35) and (22.5 <= t < 35):
                correct += 1
            elif (35 <= p < 45.5) and (35 <= t < 45.5):
                correct += 1
            elif (45.5 <= p < 56.5) and (45.5 <= t < 56.5):
                correct += 1
            elif (56.5 <= p <= 100) and (56.5 <= t <= 100):
                correct += 1
            else:
                pass
    return correct / len(target)
