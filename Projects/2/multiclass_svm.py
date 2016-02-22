import utility
from sklearn.datasets import fetch_20newsgroups
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from sklearn import metrics

categories = ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'misc.forsale', 'soc.religion.christian']
docs_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
docs_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)

svm_basic = SVC(kernel='linear', class_weight='balanced', probability=True, random_state=40)
svm_onerest = OneVsRestClassifier(svm_basic)
pipeline_svm_onerest = utility.pipeline_setup(svm_onerest)
pipeline_svm_fitted = pipeline_svm_onerest.fit(docs_train.data, docs_train.target)
svm_predict = pipeline_svm_fitted.predict(docs_test.data)
utility.print_stats(docs_test.target, svm_predict, 'SVM OneVSOne')


svm_weighted = SVC(kernel='linear', class_weight='balanced', probability=True,random_state=40) #balanced param to make sure both docs have same no. of samples in onevsone
svm_oneone = OneVsOneClassifier(svm_weighted)
pipeline_svm_oneone  = utility.pipeline_setup(svm_oneone)
pipeline_svm_fitted = pipeline_svm_oneone.fit(docs_train.data, docs_train.target)
svm_predict = pipeline_svm_fitted.predict(docs_test.data)
utility.print_stats(docs_test.target, svm_predict, 'SVM OneVSRest')