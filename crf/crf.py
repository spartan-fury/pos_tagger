import pickle
import nltk
import string
import numpy as np
from tqdm.auto import tqdm
from collections import Counter

nltk.download("punkt")
nltk.download("punkt_tab")


def process_inputs(sentences, split=False):
    crf_feats = CRFFeatures()
    processed = []
    for sentence in tqdm(sentences, desc="Processing Input data.."):
        if not split:
            sentence = tokenize(sentence)
        processed.append(crf_feats.extract_features(sentence))
    return processed


def tokenize(text):
    tokens = nltk.tokenize.word_tokenize(text)
    # tokens = [token.lower() for token in tokens]
    return tokens


def save_crf(model, save_path):
    with open(save_path, "wb") as handler:
        pickle.dump(model, handler)


def load_crf(path):
    model = None
    with open(path, "rb") as handler:
        model = pickle.load(handler)
    return model


class CRFFeatures:

    def is_punctuation(self, word):
        return all(char in string.punctuation for char in word)

    def stem_word(self, word):
        vowels = "aeiouAEIOUgyn"
        stem = word.rstrip(vowels)
        if len(stem) < 2:
            return word[:2]
        return stem

    def word2features(self, sentence, idx):
        word = sentence[idx]

        features = {
            "word": word,
            "word.lower": word.lower(),
            "prefix1": word[:1],
            "prefix2": word[:2],
            "prefix3": word[:3],
            "suffix1": word[-1:],
            "suffix2": word[-2:],
            "suffix3": word[-3:],
            "is_digit": word.isdigit(),
            "is_punct": self.is_punctuation(word),
            "word_length": len(word),
            "stem": self.stem_word(word),
        }

        if idx == 0:
            features["BOS"] = True
        elif idx == len(sentence) - 1:
            features["EOS"] = True

        if idx > 0:
            word1 = sentence[idx - 1]
            features.update(
                {
                    "-1:word": word1,
                    "-1:word.lower": word1.lower(),
                    "-1:prefix1": word1[:1],
                    "-1:prefix2": word1[:2],
                    "-1:prefix3": word1[:3],
                    "-1:suffix1": word1[-1:],
                    "-1:suffix2": word1[-2:],
                    "-1:suffix3": word1[-3:],
                    "-1:is_digit": word1.isdigit(),
                    "-1:is_punct": self.is_punctuation(word1),
                    "-1:word_length": len(word1),
                    "-1:stem": self.stem_word(word1),
                }
            )
        else:
            features["BOS"] = True

        if idx < len(sentence) - 1:
            word1 = sentence[idx + 1]
            features.update(
                {
                    "+1:word": word1,
                    "+1:word.lower": word1.lower(),
                    "+1:prefix1": word1[:1],
                    "+1:prefix2": word1[:2],
                    "+1:prefix3": word1[:3],
                    "+1:suffix1": word1[-1:],
                    "+1:suffix2": word1[-2:],
                    "+1:suffix3": word1[-3:],
                    "+1:is_digit": word1.isdigit(),
                    "+1:is_punct": self.is_punctuation(word1),
                    "+1:word_length": len(word1),
                    "+1:stem": self.stem_word(word1),
                }
            )
        else:
            features["EOS"] = True

        if idx > 1:
            word2 = sentence[idx - 2]
            features.update(
                {
                    "-2:word": word2,
                    "-2:word.lower": word2.lower(),
                    "-2:prefix1": word2[:1],
                    "-2:prefix2": word2[:2],
                    "-2:prefix3": word2[:3],
                    "-2:suffix1": word2[-1:],
                    "-2:suffix2": word2[-2:],
                    "-2:suffix3": word2[-3:],
                    "-2:is_digit": word2.isdigit(),
                    "-2:is_punct": self.is_punctuation(word2),
                    "-2:word_length": len(word2),
                    "-2:stem": self.stem_word(word2),
                }
            )

        if idx < len(sentence) - 2:
            word2 = sentence[idx + 2]
            features.update(
                {
                    "+2:word": word2,
                    "+2:word.lower": word2.lower(),
                    "+2:prefix1": word2[:1],
                    "+2:prefix2": word2[:2],
                    "+2:prefix3": word2[:3],
                    "+2:suffix1": word2[-1:],
                    "+2:suffix2": word2[-2:],
                    "+2:suffix3": word2[-3:],
                    "+2:is_digit": word2.isdigit(),
                    "+2:is_punct": self.is_punctuation(word2),
                    "+2:word_length": len(word2),
                    "+2:stem": self.stem_word(word2),
                }
            )

        return features

    def extract_features(self, sentence):
        return [self.word2features(sentence, i) for i in range(len(sentence))]
