from src.hmm import HMM, process_inputs, save_hmm
from src.data import load_brown_corpus
from src.metrics import accuracy

trainX, trainY, testX, testY = load_brown_corpus(train_size=0.9, tagset="universal")

print("Number of samples in train and test respectively: ", len(trainX), len(testX))

trainX = process_inputs(trainX, split=True)
testX = process_inputs(testX, split=True)

model = HMM()

model.fit(X=trainX, y=trainY)
save_hmm(model, "hmm.pkl")

train_predictions = model.predict(trainX)
print("Train Accuracy: ", accuracy(train_predictions, trainY))

test_predictions = model.predict(testX)
print("Test Accuracy: ", accuracy(test_predictions, testY))
