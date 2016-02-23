import string
from sklearn import feature_extraction
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from sklearn import metrics
import re

__stemmer = nltk.stem.LancasterStemmer()
__words_only = re.compile("^[A-Za-z]*$")


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
            #print w
            if w is not None and stop_word_cleaner(w) and punctuation_cleaner(w) and regex_filter(w):
                result += " " + stem_cleaner(w)
    #print result
    return result


def regex_filter(s):
    if __words_only.match(s) is not None:
        return True
    return False


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
    
    #--- for testing purposes use a small dataset ---
    #categories = ['comp.graphics', 'rec.sport.hockey']
    #data_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
    #data_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
      
    #return data_train, data_test


def draw_roc_curve(y_true, y_score):
    # Compute fpr, tpr, thresholds and roc auc
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate or (1 - Specifity)')
    plt.ylabel('True Positive Rate or (Sensitivity)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


def print_stats(expected, predicted, learning_algo):
    # summarize the fit of the model
    print('Classification report for: %s\n'%learning_algo)
    print(metrics.classification_report(expected, predicted))
    print('Confusion matrix for: %s\n'%learning_algo)
    print(metrics.confusion_matrix(expected, predicted))
    print('Accuracy for: %s\n'%learning_algo)
    print(metrics.accuracy_score(expected, predicted))
    print '\n'