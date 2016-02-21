import string
from sklearn import feature_extraction
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups

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


def custom_2class_classifier():
    categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
                  'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
    data_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
    data_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)

    class Dummy:
        def __init__(self, _data, _target):
            self.data = _data
            self.target = _target

    new_train_target = []
    for i in data_train.target:
        if i>3:
            new_train_target.append(1)
        else:
            new_train_target.append(0)

    new_train = Dummy(data_train.data, new_train_target)

    new_test_target = []
    for i in data_test.target:
        if i>3:
            new_test_target.append(1)
        else:
            new_test_target.append(0)

    new_test = Dummy(data_test.data, new_test_target)

    return new_train, new_test
