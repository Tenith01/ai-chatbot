import nltk
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
def tokenize (sentence):
    return nltk.word_tokenize(sentence)

def Stem (word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_word):
    pass

# a = "How long does shipping take ?"
# print(a)
# a = tokenize(a)
# print(a)

words = ["Organize","organizes","organizing"]
stemmed_word = [Stem(w) for w in words]
print(stemmed_word)