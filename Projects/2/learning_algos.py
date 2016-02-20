import utility
from sklearn.datasets import fetch_20newsgroups

# Load the dataset
docs_train = fetch_20newsgroups(subset='train',shuffle=True,random_state=42)
docs_test = fetch_20newsgroups(subset='test',shuffle=True,random_state=42)

learning_algo = ''
pipeline_transformed = utility.pipeline_setup(learning_algo)
print pipeline_transformed.shape

#SVM ->
# svm = SVC(kernel='linear', probability=True, random_state=40)
# svm_transformed = svm.fit(docs_train.data, docs_train.target)
# predict = svm_transformed.predict(docs_test.data)
#not sure about confusion matrix and roc curve