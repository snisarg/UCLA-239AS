import CSVtoArray
from sklearn.cross_validation import KFold

kf = KFold(18580, 10, True)

values = []

for i in range(18580):
    values.insert(i, False)

for train, test in kf:
    print (train, test)
    print (train.shape, test.shape)
    for i in test:
        values[i] = True

for i in range(len(values)):
    if values[i] is False:
        print i

