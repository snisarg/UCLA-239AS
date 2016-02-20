import string
import sklearn
import nltk

stemmer = nltk.stem.LancasterStemmer()


def punctuation_cleaner(s):
    if s not in string.punctuation:
        return True
    return False


def stop_word_cleaner(s):
    if s not in sklearn.feature_extraction.text.ENGLISH_STOP_WORDS:
        return True
    return False


def stem_cleaner(s):
    return stemmer.stem(s)


def clean_word(s):
    result = ""
    if s is not None:
        for w in nltk.tokenize.word_tokenize(s.lower()):
            if w is not None and stop_word_cleaner(w) and punctuation_cleaner(w):
                result += " " + stem_cleaner(w)
    return result


