import numpy as np
from src.hmm import HMM, process_inputs, save_hmm
from src.data import load_kfold_brown_corpus
from src.metrics import accuracy


num_folds = 5
splits = load_kfold_brown_corpus(n_splits=num_folds, tagset="universal")
train_accuracies = []
test_accuracies = []
for fold, (trainX, trainY, testX, testY) in enumerate(splits, start=1):
    print(f"Training and Evaluating: Fold {fold}..")
    print("Number of samples in train and test respectively: ", len(trainX), len(testX))

    trainX = process_inputs(trainX, split=True)
    testX = process_inputs(testX, split=True)

    model = HMM()

    model.fit(X=trainX, y=trainY)
    save_hmm(model, f"hmm_fold_{fold}.pkl")

    train_predictions = model.predict(trainX)
    train_accuracy = accuracy(train_predictions, trainY)
    train_accuracies.append(train_accuracy)
    print("Train Accuracy: ", train_accuracy)

    test_predictions = model.predict(testX)
    test_accuracy = accuracy(test_predictions, testY)
    test_accuracies.append(test_accuracy)
    print("Test Accuracy: ", test_accuracy)

print(f"{num_folds}-fold train accuracy: {np.mean(train_accuracies)}")
print(f"{num_folds}-fold cross validation accuracy: {np.mean(test_accuracies)}")
