import numpy as np
from sklearn_crfsuite import CRF
from src.crf import process_inputs, save_crf
from src.data import load_kfold_brown_corpus
from src.metrics import accuracy, compute_metrics


num_folds = 5

splits = load_kfold_brown_corpus(n_splits=num_folds, tagset="universal")

labels = [
    "PRT",
    "NUM",
    ".",
    "DET",
    "ADP",
    "X",
    "ADV",
    "VERB",
    "NOUN",
    "ADJ",
    "PRON",
    "CONJ",
]
tag2index = {tag: id for id, tag in enumerate(labels)}

train_accuracies = []
test_accuracies = []

train_precision_macro = []
train_recall_macro = []
train_f1_macro = []
train_f0_5_macro = []
train_f2_macro = []

train_precision_micro = []
train_recall_micro = []
train_f1_micro = []
train_f0_5_micro = []
train_f2_micro = []

train_precision_weighted = []
train_recall_weighted = []
train_f1_weighted = []
train_f0_5_weighted = []
train_f2_weighted = []

test_precision_macro = []
test_recall_macro = []
test_f1_macro = []
test_f0_5_macro = []
test_f2_macro = []

test_precision_micro = []
test_recall_micro = []
test_f1_micro = []
test_f0_5_micro = []
test_f2_micro = []

test_precision_weighted = []
test_recall_weighted = []
test_f1_weighted = []
test_f0_5_weighted = []
test_f2_weighted = []

test_tag_precision = []
test_tag_recall = []
test_tag_f1 = []

for fold, (trainX, trainY, testX, testY) in enumerate(splits, start=1):
    print(f"Training and Evaluating: Fold {fold}..")
    print("Number of samples in train and test respectively: ", len(trainX), len(testX))

    trainX = process_inputs(trainX, split=True)
    testX = process_inputs(testX, split=True)

    model = CRF(
        algorithm="lbfgs",
        c1=0.25,
        c2=0.3,
        max_iterations=100,
        all_possible_transitions=True,
    )

    model.fit(trainX, trainY)
    save_crf(model, f"crf_fold_{fold}.pkl")

    train_predictions = model.predict(trainX)
    train_accuracy = accuracy(train_predictions, trainY)
    metrics_1 = compute_metrics(train_predictions, trainY, labels, beta=1)
    metrics_2 = compute_metrics(train_predictions, trainY, labels, beta=2)
    metrics_0_5 = compute_metrics(train_predictions, trainY, labels, beta=0.5)
    train_accuracies.append(train_accuracy)
    train_precision_macro.append(metrics_1["precision_macro"])
    train_precision_micro.append(metrics_1["precision_micro"])
    train_precision_weighted.append(metrics_1["precision_weighted"])
    train_recall_macro.append(metrics_1["recall_macro"])
    train_recall_micro.append(metrics_1["recall_micro"])
    train_recall_weighted.append(metrics_1["recall_weighted"])
    train_f1_macro.append(metrics_1["fbeta_macro"])
    train_f1_micro.append(metrics_1["fbeta_micro"])
    train_f1_weighted.append(metrics_1["fbeta_weighted"])
    train_f0_5_macro.append(metrics_0_5["fbeta_macro"])
    train_f0_5_micro.append(metrics_0_5["fbeta_micro"])
    train_f0_5_weighted.append(metrics_0_5["fbeta_weighted"])
    train_f2_macro.append(metrics_2["fbeta_macro"])
    train_f2_micro.append(metrics_2["fbeta_micro"])
    train_f2_weighted.append(metrics_2["fbeta_weighted"])
    print("Train Accuracy: ", train_accuracy)

    test_predictions = model.predict(testX)
    test_accuracy = accuracy(test_predictions, testY)
    metrics_1 = compute_metrics(test_predictions, testY, labels, beta=1)
    metrics_2 = compute_metrics(test_predictions, testY, labels, beta=2)
    metrics_0_5 = compute_metrics(test_predictions, testY, labels, beta=0.5)
    test_accuracies.append(test_accuracy)
    test_precision_macro.append(metrics_1["precision_macro"])
    test_precision_micro.append(metrics_1["precision_micro"])
    test_precision_weighted.append(metrics_1["precision_weighted"])
    test_recall_macro.append(metrics_1["recall_macro"])
    test_recall_micro.append(metrics_1["recall_micro"])
    test_recall_weighted.append(metrics_1["recall_weighted"])
    test_f1_macro.append(metrics_1["fbeta_macro"])
    test_f1_micro.append(metrics_1["fbeta_micro"])
    test_f1_weighted.append(metrics_1["fbeta_weighted"])
    test_f0_5_macro.append(metrics_0_5["fbeta_macro"])
    test_f0_5_micro.append(metrics_0_5["fbeta_micro"])
    test_f0_5_weighted.append(metrics_0_5["fbeta_weighted"])
    test_f2_macro.append(metrics_2["fbeta_macro"])
    test_f2_micro.append(metrics_2["fbeta_micro"])
    test_f2_weighted.append(metrics_2["fbeta_weighted"])
    test_tag_precision.append(metrics_1["precision"].tolist())
    test_tag_recall.append(metrics_1["recall"].tolist())
    test_tag_f1.append(metrics_1["fbeta"].tolist())
    print("Test Accuracy: ", test_accuracy)

print(f"Overall Performance on {num_folds}-fold")
print("Train")
print(f"accuracy: {np.mean(train_accuracies)}")
print(f"precision macro: {np.mean(train_precision_macro)}")
print(f"precision micro: {np.mean(train_precision_micro)}")
print(f"precision weighted: {np.mean(train_precision_weighted)}")
print(f"recall macro: {np.mean(train_recall_macro)}")
print(f"recall micro: {np.mean(train_recall_micro)}")
print(f"recall weighted: {np.mean(train_recall_weighted)}")
print(f"f1 macro: {np.mean(train_f1_macro)}")
print(f"f1 micro: {np.mean(train_f1_micro)}")
print(f"f1 weighted: {np.mean(train_f1_weighted)}")
print(f"f0_5 macro: {np.mean(train_f0_5_macro)}")
print(f"f0_5 micro: {np.mean(train_f0_5_micro)}")
print(f"f0_5 weighted: {np.mean(train_f0_5_weighted)}")
print(f"f2 macro: {np.mean(train_f2_macro)}")
print(f"f2 micro: {np.mean(train_f2_micro)}")
print(f"f2 weighted: {np.mean(train_f2_weighted)}")
print("...")
print("Test")
print(f"accuracy: {np.mean(test_accuracies)}")
print(f"precision macro: {np.mean(test_precision_macro)}")
print(f"precision micro: {np.mean(test_precision_micro)}")
print(f"precision weighted: {np.mean(test_precision_weighted)}")
print(f"recall macro: {np.mean(test_recall_macro)}")
print(f"recall micro: {np.mean(test_recall_micro)}")
print(f"recall weighted: {np.mean(test_recall_weighted)}")
print(f"f1 macro: {np.mean(test_f1_macro)}")
print(f"f1 micro: {np.mean(test_f1_micro)}")
print(f"f1 weighted: {np.mean(test_f1_weighted)}")
print(f"f0_5 macro: {np.mean(test_f0_5_macro)}")
print(f"f0_5 micro: {np.mean(test_f0_5_micro)}")
print(f"f0_5 weighted: {np.mean(test_f0_5_weighted)}")
print(f"f2 macro: {np.mean(test_f2_macro)}")
print(f"f2 micro: {np.mean(test_f2_micro)}")
print(f"f2 weighted: {np.mean(test_f2_weighted)}")
test_tag_precision = np.mean(test_tag_precision, axis=0)
test_tag_recall = np.mean(test_tag_recall, axis=0)
test_tag_f1 = np.mean(test_tag_f1, axis=0)
print("...")
print("Class Wise Metrics")
print(".....")
print("Precision")
for tag, idx in tag2index.items():
    print(f"{tag}: {test_tag_precision[idx]}")
print(".....")
print("Recall")
for tag, idx in tag2index.items():
    print(f"{tag}: {test_tag_recall[idx]}")
print(".....")
print("F1")
for tag, idx in tag2index.items():
    print(f"{tag}: {test_tag_f1[idx]}")
