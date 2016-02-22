from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import utility
import math
import pickle

# returns top 'n' significant terms for the given category
def get_significant_terms(category):

    print "in get_significant terms()"
    all_classes = list(fetch_20newsgroups(subset='train').target_names)
    index = all_classes.index(category)
    freq_matrix = term_class_count()
    f = open('freq_matrix.pk1','wb')
    pickle.dump(freq_matrix, f)
    f.close()

    print "frequency matrix done"
    class_max_freq = max(freq_matrix[index])
    class_count_list = []
    class_count_list = get_term_class_count(freq_matrix)
    f = open('class_count_list.pk1','wb')
    pickle.dump(class_count_list , f)
    f.close()

    category_list = []
    category_list.append(category)
    class_freq_list = []
    names, class_freq_list = get_class_frequency(category_list) # get keyword list &  normal freq  list per class

    f = open('class_freq_list.pk1','wb')
    pickle.dump(class_freq_list, f)
    f.close()


    print "class max freq"
    print class_max_freq
    print "class_freq_list"
    print class_freq_list
    print "class_count_list"
    print class_count_list
    print "computing formula"
    tf_icf_list = []
    # log base 2 in calculation
    for i in range(len(class_freq_list)):
        #print class_freq_list[i]
        value = ((0.5 + (0.5 * ( class_freq_list[i]) / class_max_freq))) * (math.log((20/class_count_list[i] ), 2))
        print "value" + value
        tf_icf_list.append(value)

    top_ten_indexes = []
    top_ten_indexes = sorted(range(len(tf_icf_list)), key=lambda i: tf_icf_list[i])[-10:]

    keyword_list = []
    for i in range(len(top_ten_indexes)):
        keyword_list.append(names[top_ten_indexes[i]])

    return keyword_list

def get_class_frequency(category):

    category = fetch_20newsgroups(subset='train', categories=category, remove=('headers','footers','quotes'))
    count_vector = CountVectorizer(preprocessor=utility.clean_word)
    doc_term_matrix = count_vector.fit_transform(category.data)
    names = count_vector.get_feature_names()

    no_terms = doc_term_matrix.shape[1]
    class_freq_list = []

    for i in range(no_terms):
        class_freq_list.append( sum(doc_term_matrix[:,i]) )
    print "get_class_frequency() done "
    return (names, class_freq_list)

# returns matrix of frequency count of all classes
def term_class_count():

    all_classes = list(fetch_20newsgroups(subset='train').target_names)
    all_class_docs = []

    for doc_class in all_classes:
        category = fetch_20newsgroups(subset='train', categories=[doc_class], remove=('headers','footers','quotes'))
        docs_list = category.data
        per_class_docs = ''

        for doc in docs_list:
            clean_doc = utility.clean_word(doc)
            per_class_docs += ''+clean_doc

        all_class_docs.append(per_class_docs)

    count_vector = CountVectorizer()
    freq_matrix = count_vector.fit_transform(all_class_docs)
    print freq_matrix.shape
    print freq_matrix

    return freq_matrix


    '''
    f = open('output.txt', 'w')
    class_docs = fetch_20newsgroups(subset='train', shuffle=True, random_state=42) #load docs of current class
    count_vector = CountVectorizer(preprocessor=utility.clean_word)
    freq_matrix = count_vector.fit_transform(class_docs.data)

    indexes_lists = freq_matrix.nonzero()[0]
    s = str(indexes_lists)
    f.write(s)
    #print indexes_lists
    print 'done'
    '''

    #concatenate all files in a directory
'''
    #path = '..\..\Datasets\Newsgroup'
    path = 'C:/Users/ashish/PycharmProjects/UCLA-239AS/Datasets/Newsgroup/'
  #  print path

    for subdirs, dirs, files in os.walk(path):
        print 'subdirs are'
        print dirs

        filename = subdirs
        print filename
        outputfile = open(filename,'a')
        for file in files:
            print 'file is : '
            print file
            outputfile.write(file.read( ))
            file.close()

        outputfile.close()
    print 'done'
'''





'''
# denominator of log term in tf-icf calculation
# returns an array in which each val is the number of classes in which it appears
def term_class_count(n_cols):

    classes = ['alt.atheism', 'comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'comp.windows.x', 'misc.forsale', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey', 'sci.crypt', 'sci.electronics', 'sci.med', 'sci.space', 'soc.religion.christian', 'talk.politics.guns', 'talk.politics.mideast', 'talk.politics.misc', 'talk.religion.misc']
    output = [0] * n_cols   # counter list

    for doc_class in classes:

        doc_class_list = []
        doc_class_list.append(doc_class)
        class_docs = fetch_20newsgroups(subset='train', categories=doc_class_list, shuffle=True, random_state=42) #load docs of current class
        count_vector = CountVectorizer(preprocessor=utility.clean_word)
        class_term_freq_matrix = count_vector.fit_transform(class_docs.data)  #matrix of raw frequencies

        cols = class_term_freq_matrix.shape[1]
        term_per_class_freq_list = []

        # 1 output list for each class in the list
        for i in range(cols):
            term_per_class_freq_list[i] = sum(class_term_freq_matrix[:, i])
            if(term_per_class_freq_list[i] != 0):
                output[i] += 1

    return
'''


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
n = 10
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
    terms = get_significant_terms(category)
    print category
    print terms

