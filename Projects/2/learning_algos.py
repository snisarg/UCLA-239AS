from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import SVC
import utility
from sklearn.pipeline import Pipeline

# Load the dataset
docs_train = fetch_20newsgroups(subset='train',shuffle=True,random_state=42)
docs_test = fetch_20newsgroups(subset='test',shuffle=True,random_state=42)

tf_idf = TfidfVectorizer(preprocessor=utility.clean_word, use_idf=True)

#get Tf-idf-weighted document-term matrix
#rows represent documents, columns represent terms, each row is a vector for a document
#tf_idf_data = tf_idf.fit_transform(docs_train.data)

#LSA ->
lsa = TruncatedSVD(n_components=50, n_iter=5, random_state=25)
#lsa_transformed = lsa.fit_transform(lsa.data) #not sure about this part

pipeline = Pipeline([('tfidf', tf_idf),
                            ('svd', lsa)])

pipeline_transformed = pipeline.fit_transform(docs_train.data)
print pipeline_transformed.shape

#SVM ->
svm = SVC(kernel='linear', probability=True, random_state=40)
svm_transformed = svm.fit(docs_train.data, docs_train.target)
predict = svm_transformed.predict(docs_test.data)
#not sure about confusion matrix and roc curve