from sklearn_crfsuite import CRF
from src.crf import process_inputs, save_crf
from src.data import load_brown_corpus
from src.metrics import accuracy

trainX, trainY, testX, testY = load_brown_corpus(train_size=0.9, tagset="universal")

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
save_crf(model, "crf.pkl")

train_predictions = model.predict(trainX)
print("Train Accuracy: ", accuracy(train_predictions, trainY))

test_predictions = model.predict(testX)
print("Test Accuracy: ", accuracy(test_predictions, testY))
