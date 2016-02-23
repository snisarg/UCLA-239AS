import utility
import numpy
from sklearn.datasets import fetch_20newsgroups
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
# Load the dataset
docs_train, docs_test = utility.custom_2class_classifier()

#SVM ->
svm = SVC(kernel='linear', probability=True, random_state=40)
pipeline_svm = utility.pipeline_setup(svm) #pipeline_svm obj to be used in all svm algos
pipeline_svm_fitted = pipeline_svm.fit(docs_train.data, docs_train.target)
svm_predict = pipeline_svm_fitted.predict(docs_test.data)
utility.print_stats(docs_test.target, svm_predict, 'SVM Normal')
utility.draw_roc_curve(docs_test.target, pipeline_svm_fitted.predict_proba(docs_test.data)[:, 1])

#Soft margin SVM ->
#confirm this part, not sure of any other way to implement soft margin SVM
params = {
    'learning_algo__gamma': [1e-3, 1e3] #10^-3 to 10^3
}
svm_soft_margin = GridSearchCV(pipeline_svm, params, cv=5)
svm_soft_margin_fitted = svm_soft_margin.fit(docs_train.data, docs_train.target)
svm_soft_margin_predict = svm_soft_margin_fitted.predict(docs_test.data)
utility.print_stats(docs_test.target, svm_soft_margin_predict, 'Soft Margin SVM')
utility.draw_roc_curve(docs_test.target, svm_soft_margin_fitted.predict_proba(docs_test.data)[:, 1])

best_params = svm_soft_margin.best_estimator_.get_params()
for param_name in sorted(params.keys()):
    print("\t{}: {}".format(param_name, best_params[param_name]))
              
#Logistic Regression ->
logistic_regr = LogisticRegression(penalty='l2', max_iter=5, random_state=40)
pipeline_regr = utility.pipeline_setup(logistic_regr)
pipeline_regr_fitted = pipeline_regr.fit(docs_train.data, docs_train.target)
regr_predict = pipeline_regr_fitted.predict(docs_test.data)
utility.print_stats(docs_test.target, regr_predict, 'Logistic Regression')
utility.draw_roc_curve(docs_test.target, pipeline_regr_fitted.predict_proba(docs_test.data)[:, 1])