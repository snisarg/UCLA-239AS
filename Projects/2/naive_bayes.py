from sklearn.datasets import fetch_20newsgroups
import matplotlib.pyplot as pyplot
from sklearn.naive_bayes import GaussianNB
import utility

data_train, data_test = utility.custom_2class_classifier()

model = utility.pipeline_setup(GaussianNB())
model.fit(data_train.data, data_train.target)
print(model)
# make predictions
expected = data_test.target
predicted = model.predict(data_test.data)

utility.print_stats(expected, predicted)

utility.draw_roc_curve(expected, model.predict_proba(data_test.data)[:, 1])
