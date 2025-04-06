import numpy as np
import sklearn.metrics as metrics


def accuracy(y_pred, y_true):
    correct = total = 0
    for pred, true in zip(y_pred, y_true):
        correct += sum(p == t for p, t in zip(pred, true))
        total += len(true)
    accuracy = correct / (total + 1e-20)
    return accuracy


def tagwise_accuracy(y_pred, y_true, tag2index, index2tag):
    correct = np.zeros(len(index2tag))
    total = np.zeros(len(index2tag))
    for pred, true in zip(y_pred, y_true):
        for p, t in zip(pred, true):
            if p == t:
                correct[tag2index[t]] += 1
            total[tag2index[t]] += 1
    accuracy = correct / (total + 1e-20)
    return_dict = {index2tag[i]: float(acc) for i, acc in enumerate(accuracy)}
    return_dict.pop("<start>")
    return_dict.pop("<end>")
    return return_dict


def confusion_matrix(y_pred, y_true, tag2index):
    y_pred = [tag2index[y] for preds in y_pred for y in preds]
    y_true = [tag2index[y] for trues in y_true for y in trues]
    cm = metrics.confusion_matrix(y_true, y_pred, labels=list(range(len(tag2index))))
    return cm[2:, 2:]
