import numpy as np
import sklearn.metrics as metrics


def accuracy(y_pred, y_true):
    correct = total = 0
    for pred, true in zip(y_pred, y_true):
        correct += sum(p == t for p, t in zip(pred, true))
        total += len(true)
    accuracy = correct / (total + 1e-20)
    return accuracy


def tagwise_accuracy(y_pred, y_true, labels):
    tag2index = {tag: id for id, tag in enumerate(labels)}
    correct = np.zeros(len(tag2index))
    total = np.zeros(len(tag2index))
    for pred, true in zip(y_pred, y_true):
        for p, t in zip(pred, true):
            if p == t:
                correct[tag2index[t]] += 1
            total[tag2index[t]] += 1
    accuracy = correct / (total + 1e-20)
    return_dict = {labels[i]: float(acc) for i, acc in enumerate(accuracy)}
    return return_dict


def confusion_matrix(y_pred, y_true, labels):
    tag2index = {tag: id for id, tag in enumerate(labels)}
    y_pred = [tag2index[y] for preds in y_pred for y in preds]
    y_true = [tag2index[y] for trues in y_true for y in trues]
    cm = metrics.confusion_matrix(y_true, y_pred, labels=list(range(len(tag2index))))
    return cm


def compute_metrics(y_pred, y_true, labels, beta=1.0):
    tag2index = {tag: id for id, tag in enumerate(labels)}
    labels = list(tag2index.values())

    y_pred = [tag2index[y] for preds in y_pred for y in preds]
    y_true = [tag2index[y] for trues in y_true for y in trues]

    precision = metrics.precision_score(
        y_true, y_pred, average=None, labels=labels, zero_division=0
    )
    recall = metrics.recall_score(
        y_true, y_pred, average=None, labels=labels, zero_division=0
    )
    fbeta = metrics.fbeta_score(
        y_true, y_pred, beta=beta, average=None, labels=labels, zero_division=0
    )

    precision_macro = metrics.precision_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    recall_macro = metrics.recall_score(
        y_true, y_pred, average="macro", zero_division=0
    )
    fbeta_macro = metrics.fbeta_score(
        y_true, y_pred, beta=beta, average="macro", zero_division=0
    )

    precision_micro = metrics.precision_score(
        y_true, y_pred, average="micro", zero_division=0
    )
    recall_micro = metrics.recall_score(
        y_true, y_pred, average="micro", zero_division=0
    )
    fbeta_micro = metrics.fbeta_score(
        y_true, y_pred, beta=beta, average="micro", zero_division=0
    )

    precision_weighted = metrics.precision_score(
        y_true, y_pred, average="weighted", zero_division=0
    )
    recall_weighted = metrics.recall_score(
        y_true, y_pred, average="weighted", zero_division=0
    )
    fbeta_weighted = metrics.fbeta_score(
        y_true, y_pred, beta=beta, average="weighted", zero_division=0
    )

    tag_metrics = {}
    for tag, idx in tag2index.items():
        tag_metrics[tag] = {
            "precision": precision[idx],
            "recall": recall[idx],
            "fbeta": fbeta[idx],
        }

    metrics_dict = {
        "precision": precision,
        "recall": recall,
        "fbeta": fbeta,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "fbeta_macro": fbeta_macro,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "fbeta_micro": fbeta_micro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "fbeta_weighted": fbeta_weighted,
        "classwise_metrics": tag_metrics,
    }

    return metrics_dict


def find_wrong_predictions(pred_preds, true_preds, incorrect_cases, testX):
    wrong_predictions = []
    for i, (true_seq, pred_seq) in enumerate(zip(true_preds, pred_preds)):
        for j, (true_tag, pred_tag) in enumerate(zip(true_seq, pred_seq)):
            if (true_tag, pred_tag) in incorrect_cases:
                wrong_predictions.append(
                    {
                        "sentence_index": i,
                        "word_index": j,
                        "true_tag": true_tag,
                        "predicted_tag": pred_tag,
                        "word": testX[i][j],
                    }
                )
    return wrong_predictions
