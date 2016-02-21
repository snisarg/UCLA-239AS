import utility
from sklearn.datasets import fetch_20newsgroups
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
# Load the dataset
docs_train, docs_test = utility.custom_2class_classifier()

#docs_train = fetch_20newsgroups(subset='train', shuffle=True,random_state=42)
#docs_test = fetch_20newsgroups(subset='test',shuffle=True,random_state=42)

# learning_algo = ''
# pipeline_transformed = utility.pipeline_setup(learning_algo)
# print pipeline_transformed.shape

#SVM ->
svm = SVC(kernel='linear', probability=True, random_state=40)
pipeline_svm = utility.pipeline_setup(svm) #pipeline_svm obj to be used in all svm algos

pipeline_svm_fitted = pipeline_svm.fit(docs_train.data, docs_train.target)
svm_predict = pipeline_svm_fitted.predict(docs_test.data)
utility.print_stats(docs_test.target, svm_predict, 'SVM')
utility.draw_roc_curve(docs_test.target, pipeline_svm_fitted.predict_proba(docs_test.data)[:, 1])

# print("Confusion matrix for SVM:\n%s" %metrics.confusion_matrix(docs_test.target, svm_predict))
# print("Precision-Recall metrics for SVM:\n")
# print metrics.precision_recall_fscore_support(docs_test.target, svm_predict, target_names=docs_test.target_names)
# print("Accuracy for SVM:\n%s" %metrics.accuracy_score(docs_test.target, svm_predict))


             
#Logistic Regression ->
logistic_regr = LogisticRegression(penalty='l2', max_iter=5, random_state=40)
pipeline_regr = utility.pipeline_setup(logistic_regr)
pipeline_regr_fitted = pipeline_regr.fit(docs_train.data, docs_train.target)
regr_predict = pipeline_regr_fitted.predict(docs_test.data)
utility.print_stats(docs_test.target, regr_predict, 'Logistic Regression')
utility.draw_roc_curve(docs_test.target, pipeline_regr_fitted.predict_proba(docs_test.data)[:, 1])