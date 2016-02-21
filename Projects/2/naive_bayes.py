from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as pyplot
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import utility

categories = ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'misc.forsale', 'soc.religion.christian']
data_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
data_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)

model = utility.pipeline_setup(GaussianNB())
model.fit(data_train.data, data_train.target)
print(model)
# make predictions
expected = data_test.target
predicted = model.predict(data_test.data)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))
