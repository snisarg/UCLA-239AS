from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import utility
import math
import pickle
import numpy


# returns top 'n' significant terms for the given category
def get_significant_terms(category, freq_matrix):

    print "in get_significant terms()"
    all_classes = list(fetch_20newsgroups(subset='train').target_names)
    index = all_classes.index(category)


    '''
    f = open('freq_matrix.pkl','wb')
    pickle.dump(freq_matrix, f)
    f.close()

    f = open('freq_matrix.pkl','rb')
    freq_matrix = pickle.load(f)
    f.close()
    '''
    print "frequency matrix done"

    row = freq_matrix.getrow(index)
    #class_max_freq = max(row.data)
    class_max_freq = row.max()

    print "classmaxfreq"
    print class_max_freq
    class_count_list = []
    class_count_list = get_term_class_count(freq_matrix)

    '''
    f = open('class_count_list.pkl','wb')
    pickle.dump(class_count_list , f)
    f.close()

    f = open('class_count_list.pkl','rb')
    class_count_list = pickle.load(f)
    f.close()
    '''

    category_list = []
    category_list.append(category)
    class_freq_list = []
    names, class_freq_list = get_class_frequency(category_list) # get keyword list &  normal freq  list per class
#   names = get_class_frequency(category_list) # get keyword list &  normal freq  list per class

    '''
    f = open('class_freq_list.pkl','wb')
    pickle.dump(class_freq_list, f)
    f.close()
    f = open('class_freq_list.pkl','rb')
    class_freq_list = pickle.load(f)
    f.close()
    '''
    print "class_freq_list"
    print class_freq_list
    #print "class_count_list"
    #print class_count_list
    print "computing formula"
    tf_icf_list = []
    # log base 2 in calculation
    for i in range(len(class_freq_list)):
        #print class_freq_list[i]
        value = ((0.5 + (0.5 * ( class_freq_list[i]) / class_max_freq))) * (math.log((20/class_count_list[i] ), 2))
   #     print value
        tf_icf_list.append(value)

    top_ten_indexes = []
    top_ten_indexes = sorted(range(len(tf_icf_list)), key=lambda i: tf_icf_list[i])[-10:]

    keyword_list = []
    for i in range(len(top_ten_indexes)):
        keyword_list.append(names[top_ten_indexes[i]])

    return keyword_list


def get_class_frequency(category):

    category_ob = fetch_20newsgroups(subset='train', categories=category, remove=('headers','footers','quotes'))
    count_vector = CountVectorizer(preprocessor=utility.clean_word)
    doc_term_matrix = count_vector.fit_transform(category_ob.data)
    names = count_vector.get_feature_names()

    no_terms = doc_term_matrix.shape[1]
    class_freq_list = []

    #for i in range(20):
    for i in range(no_terms):
        col = doc_term_matrix.getcol(i)
        total = col.sum()
        #print total
        #print col.data
        class_freq_list.append(total)
    print "get_class_frequency() done "
    print "class_freq_list size "
    print len(class_freq_list)

   # return names
    return (names, class_freq_list)


# returns matrix of frequency count of all classes
def term_class_count():

    all_classes = list(fetch_20newsgroups(subset='train').target_names)
    all_class_docs = []

    for doc_class in all_classes:
        category_ob = fetch_20newsgroups(subset='train', categories=[doc_class], remove=('headers','footers','quotes'))
        docs_list = category_ob.data
        per_class_docs = ''

        for doc in docs_list:
            clean_doc = utility.clean_word(doc)
            per_class_docs += ' '+clean_doc

        all_class_docs.append(per_class_docs)

    count_vector = CountVectorizer()
    freq_matrix = count_vector.fit_transform(all_class_docs)
    #print freq_matrix.shape
    #print freq_matrix

    return freq_matrix


# Preprocessing to get the no of classes in which each term appears
# output- index term, value- no of classes in which that term appears
def get_term_class_count(freq_matrix):

#    freq_matrix = term_class_count()
    no_terms = freq_matrix.shape[1]
    class_count_list = [0 for i in range(no_terms)]

    for i in range(no_terms):
        for j in range(0, 20):
            if (freq_matrix[j, i] != 0):
                class_count_list[i] += 1
    print "get_term_class_count( ) done"
    return class_count_list


categories = ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'misc.forsale', 'soc.religion.christian']
freq_matrix = term_class_count()

for category in categories:

    '''
    category_list = []
    category_list.append(category)

    category_docs = fetch_20newsgroups(subset='train', categories=category_list, shuffle=True, random_state=42) #load docs of current category
    count_vector = CountVectorizer(preprocessor=utility.clean_word)
    class_term_freq_matrix = count_vector.fit_transform(category_docs.data)  #matrix of raw frequencies

    rows = class_term_freq_matrix.shape[0]
    cols = class_term_freq_matrix.shape[1]

    print rows
    print cols
    category_list.remove(category)
    '''
    terms = []
    terms = get_significant_terms(category, freq_matrix)
    print category
    print terms

