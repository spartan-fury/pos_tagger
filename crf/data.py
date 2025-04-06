import nltk
import random
from sklearn.model_selection import KFold

nltk.download("brown")
nltk.download("universal_tagset")


def load_brown_corpus(train_size=0.9, tagset="universal", seed=42):
    random.seed(seed)
    corpus = list(nltk.corpus.brown.tagged_sents(tagset=tagset))
    random.shuffle(corpus)
    split_size = int(len(corpus) * train_size)
    train_data = corpus[:split_size]
    test_data = corpus[split_size:]
    train_x = [[x for x, _ in data] for data in train_data]
    train_y = [[y for _, y in data] for data in train_data]
    test_x = [[x for x, _ in data] for data in test_data]
    test_y = [[y for _, y in data] for data in test_data]
    return train_x, train_y, test_x, test_y


def load_kfold_brown_corpus(n_splits=5, tagset="universal", shuffle=False):
    corpus = list(nltk.corpus.brown.tagged_sents(tagset=tagset))
    kf = KFold(n_splits=n_splits, random_state=None, shuffle=shuffle)
    X = [[x for x, _ in data] for data in corpus]
    Y = [[y for _, y in data] for data in corpus]
    splits = []
    for train_index, test_index in kf.split(X):
        trainX = [X[i] for i in train_index]
        testX = [X[i] for i in test_index]
        trainY = [Y[i] for i in train_index]
        testY = [Y[i] for i in test_index]
        splits.append((trainX, trainY, testX, testY))
    return splits
