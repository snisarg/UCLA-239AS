import string
from sklearn import feature_extraction
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

__stemmer = nltk.stem.LancasterStemmer()


def punctuation_cleaner(s):
    if s not in string.punctuation:
        return True
    return False


def stop_word_cleaner(s):
    if s not in feature_extraction.text.ENGLISH_STOP_WORDS:
        return True
    return False


def stem_cleaner(s):
    return __stemmer.stem(s)


def clean_word(s):
    result = ""
    if s is not None:
        for w in nltk.tokenize.word_tokenize(s.lower()):
            if w is not None and stop_word_cleaner(w) and punctuation_cleaner(w):
                result += " " + stem_cleaner(w)
    return result


def pipeline_setup(learning_algo):
    tf_idf = TfidfVectorizer(preprocessor=clean_word, use_idf=True)
    lsa = TruncatedSVD(n_components=50, n_iter=5, random_state=25)
    pipeline_list = [('tf_idf', tf_idf), ('svd', lsa), ('learning_algo', learning_algo)]
    pipeline = Pipeline(pipeline_list)
    return pipeline

