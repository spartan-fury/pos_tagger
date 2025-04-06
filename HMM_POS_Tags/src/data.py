import nltk
import random
from sklearn.model_selection import KFold

nltk.download("brown")
nltk.download("universal_tagset")


def load_brown_corpus(train_size=0.9, tagset="universal", seed=42):
    random.seed(seed)
    corpus = list(nltk.corpus.brown.tagged_sents(tagset=tagset))
    #SO BASICALLY WE'RE TAKING IN THE BROWN CORPUS, AND THERES THE SECTION ALREADY WITH THE TAGGED SENTENCES, USING THE UNIVERSAL TAGSET HERE
    random.shuffle(corpus)
    split_size = int(len(corpus) * train_size)
    train_data = corpus[:split_size]
    test_data = corpus[split_size:]
    train_x = [[x for x, _ in data] for data in train_data]
    train_y = [[y for _, y in data] for data in train_data]
    test_x = [[x for x, _ in data] for data in test_data]
    test_y = [[y for _, y in data] for data in test_data]
    #AFTER A SIMPLE SPLIT, WE JSUT RETURN BACK THE TRAIN AND TEST X AND Y
    return train_x, train_y, test_x, test_y  



def load_kfold_brown_corpus(n_splits=5, tagset="universal", shuffle=False):
    corpus = list(nltk.corpus.brown.tagged_sents(tagset=tagset))
    kf = KFold(n_splits=n_splits, random_state=None, shuffle=shuffle)
    #SO HERE BASICALLY WE HAVE THE K FOLDS USED FOR THE K FOLD VALIDATION (SCIKIT LEARN FUNCTION)
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
