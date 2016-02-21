import utility
from sklearn.datasets import fetch_20newsgroups
from sklearn.svm import SVC
from sklearn import metrics

# Load the dataset
docs_train = fetch_20newsgroups(subset='train', shuffle=True,random_state=42)
docs_test = fetch_20newsgroups(subset='test',shuffle=True,random_state=42)

# learning_algo = ''
# pipeline_transformed = utility.pipeline_setup(learning_algo)
# print pipeline_transformed.shape

#SVM ->
svm = SVC(kernel='linear', probability=True, random_state=40)
pipeline = utility.pipeline_setup(svm)
pipeline = pipeline.fit(docs_train.data, docs_train.target)
svm_predict = pipeline.predict(docs_test.data)
#svm_pred_proba = pipeline.predict_proba(docs_test.data)

print("Confusion matrix for SVM:\n%s" %metrics.confusion_matrix(docs_test.target, svm_predict))
print("Precision-Recall metrics for SVM:\n")
print metrics.precision_recall_fscore_support(docs_test.target, svm_predict, target_names=docs_test.target_names)
print("Accuracy for SVM:\n%s" %metrics.accuracy_score(docs_test.target, svm_predict))
