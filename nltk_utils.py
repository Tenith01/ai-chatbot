import nltk
import numpy as np
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()
nltk.download('punkt')



def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, all_word):
    """
        return bag of words array:
        1 for each known word that exists in the sentence, 0 otherwise
        example:
        sentence = ["hello", "how", "are", "you"]
        words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
        bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    tokenized_sentence = [stem(w) for w in tokenized_sentence]

    bag = np.zeros(len(all_word), dtype=np.float32)
    for idx, w in enumerate(all_word):
        if w in tokenized_sentence:
            bag[idx] = 1.0

    return bag


# sentence = ["hello", "how", "are", "you"]
# words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
# bog = bag_of_words(sentence, words)
# print(bog)


# a = "How long does shipping take ?"
# print(a)
# a = tokenize(a)
# print(a)

# words = ["Organize","organizes","organizing"]
# stemmed_word = [stem(w) for w in words]
# print(stemmed_word)
