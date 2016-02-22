from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as pyplot
from sklearn.naive_bayes import GaussianNB
import utility

docs_train, docs_test = utility.custom_2class_classifier()

model = utility.pipeline_setup(GaussianNB())
model_fitted = model.fit(docs_train.data, docs_train.target)
#print(model)
# make predictions
expected = docs_test.target
predicted = model_fitted.predict(docs_test.data)
utility.print_stats(expected, predicted, 'Naive Bayes Basic')
utility.draw_roc_curve(expected, model_fitted.predict_proba(docs_test.data)[:, 1])