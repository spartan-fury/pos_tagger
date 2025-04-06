import pickle
import nltk
import numpy as np
from tqdm.auto import tqdm
from collections import Counter

nltk.download("punkt")
nltk.download("punkt_tab")


def process_inputs(sentences, split=False):
    processed = []
    for sentence in tqdm(sentences, desc="Processing Input data.."):
        if split:
            processed.append([word.lower().strip() for word in sentence])
        else:
            processed.append(tokenize(sentence))
    return processed


def tokenize(text):
    tokens = nltk.tokenize.word_tokenize(text)
    tokens = [token.lower() for token in tokens]
    return tokens


def save_hmm(model, save_path):
    with open(save_path, "wb") as handler:
        pickle.dump(model, handler)


def load_hmm(path):
    model = None
    with open(path, "rb") as handler:
        model = pickle.load(handler)
    return model



#BASICALLY THE DEAL IS TO HAVE THE POS TAGGINGS TO BE THE HIDDEN MARKOV CHAIN, AND THE PROBABILITIES OF THESE WOULD BE THE 'TRANSITION MATRIX'
#AND THE OBSERVABLE IS THE WORD OUTPUT, AND THE PROBABILITIES BETWEEN THE POS TAGGING TO THE WORD WILL BE IN THE 'EMISSION MATRIX'

class HMM:
    def __init__(self, unk_threshold=5) -> None:
        self.transition_prob = None
        self.emission_prob = None
        self.word2idx = None
        self.tag2idx = None
        self.idx2word = None
        self.idx2tag = None
        self.unk_threshold = unk_threshold
        self.unk_idx = None
        self.start_idx = 0
        self.end_idx = 1

    def __compile(self, X, y):
        words = Counter()
        tags = set()
        for sentence, target in zip(X, y):
            words.update(sentence)
            tags.update(target)
        self.word2idx = {"<unk>": 0}
        words_list = [w for w, cnt in words.items() if cnt > self.unk_threshold]   #so basically only taking those words which appear more than the threshold number of times
        self.word2idx.update({word: i for i, word in enumerate(words_list, 1)})    #numbering of the words starts from 1, since 0 is the unknown token
        self.unk_idx = self.word2idx["<unk>"]
        self.tag2idx = {"<start>": self.start_idx, "<end>": self.end_idx}
        self.tag2idx.update({tag: i for i, tag in enumerate(tags, 2)})           #the POS tags start from 2 since the 0 and 1 tag is for the start and end ids

        #reverse mapping the word and tags
        self.idx2word = {i: word for word, i in self.word2idx.items()}
        self.idx2tag = {i: tag for tag, i in self.tag2idx.items()}

    #so the point of this function is to just create the transition and emission probability matrices (this is a probabilistic model)
    def fit(self, X, y):
        self.__compile(X=X, y=y)  #basically updates the self variables nicely

        #the num of tags and words
        num_tags = len(self.tag2idx)
        num_words = len(self.word2idx)

        #initialising the probability matrices
        self.transition_prob = np.zeros((num_tags, num_tags))       #SHAPE IS BASED ON TAG TO TAG      
        self.emission_prob = np.zeros((num_tags, num_words))        #SHAPE IS BASED ON TAG TO WORD
        print(self.emission_prob.shape)

        tag_counts = np.zeros(num_tags)

        for sentence, target in tqdm(zip(X, y), total=len(X), desc="Fitting HMM.."):
            previous_tag_idx = self.start_idx
            for word, tag in zip(sentence, target):
                word_idx = self.word2idx.get(word, self.unk_idx)
                tag_idx = self.tag2idx[tag]
                self.transition_prob[previous_tag_idx][tag_idx] += 1
                self.emission_prob[tag_idx][word_idx] += 1
                tag_counts[tag_idx] += 1
                previous_tag_idx = tag_idx
            self.transition_prob[previous_tag_idx][self.end_idx] += 1

        # to make sure we are not dividing by 0 we add 1e-10
        self.transition_prob = (
            self.transition_prob.T / (self.transition_prob.sum(axis=1) + 1e-10)
        ).T
        self.emission_prob = (self.emission_prob.T / (tag_counts + 1e-10)).T

    
    def __viterbi(self, words):
        # Exclude <start> and <end>
        num_tags = len(self.tag2idx)
        V = np.zeros((num_tags, len(words)))
        backpointer = np.zeros((num_tags, len(words)), dtype=int)

        # Initial step
        for tag_idx in range(num_tags):
            word_idx = self.word2idx.get(words[0], self.unk_idx)
            V[tag_idx, 0] = (
                self.transition_prob[self.start_idx, tag_idx]
                * self.emission_prob[tag_idx, word_idx]
            )

        # Recursion step
        for t in range(1, len(words)):
            word_idx = self.word2idx.get(words[t], self.unk_idx)
            for tag_idx in range(num_tags):
                prob, prev_tag = max(
                    (
                        V[prev_tag_idx, t - 1]
                        * self.transition_prob[prev_tag_idx, tag_idx]
                        * self.emission_prob[tag_idx, word_idx],
                        prev_tag_idx,
                    )
                    for prev_tag_idx in range(num_tags)
                )
                V[tag_idx, t] = prob
                backpointer[tag_idx, t] = prev_tag

        # Termination step
        best_path = []
        last_tag = np.argmax(V[:, -1])
        best_path.append(last_tag)

        for t in range(len(words) - 1, 0, -1):
            last_tag = backpointer[last_tag, t]
            best_path.append(last_tag)

        return [self.idx2tag[int(tag_idx)] for tag_idx in reversed(best_path)]

    def predict(self, X):
        if self.emission_prob is None or self.transition_prob is None:
            raise Exception("Model is not trained yet. Please fit the model first.")
        return [self.__viterbi(words) for words in tqdm(X, desc="Predicting..")]
