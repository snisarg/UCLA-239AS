from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import utility

categories = ['comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'misc.forsale', 'soc.religion.christian']
docs_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)#, remove=('headers','footers','quotes'))
docs_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)#, remove=('headers','footers','quotes'))

model = utility.pipeline_setup(GaussianNB())
model.fit(docs_train.data, docs_train.target)
# print(model)
# make predictions
expected = docs_test.target
predicted = model.predict(docs_test.data)

utility.print_stats(expected, predicted, 'Naive Bayes Multiclass')