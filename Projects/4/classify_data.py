import pandas
from sklearn import linear_model, cross_validation, neural_network
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
import numpy

K_FOLDS = 10

def linear_regression(X, Y):
    length = len(X)

    print("\nX.shape : ", X.shape)
    print("\nY.shape : ", Y.shape)

    model = linear_model.LogisticRegression()

    # X = X.values.reshape(length, 1)
    #Y = Y.reshape(length, 1)
    Y = Y.ravel()

    predicted = cross_validation.cross_val_predict(model, X, Y, K_FOLDS, 1, 0, None, 0)
    scores = cross_validation.cross_val_score(model, X, Y, cv=K_FOLDS, scoring='mean_absolute_error')

    avg_scores = numpy.average(numpy.abs(scores))
    print("Mean absolute error : ", avg_scores)

    '''
    print 'All RMSEs',  numpy.sqrt(-scores)
    print 'Mean RMSE',  numpy.mean(numpy.sqrt(-scores))
    print 'Best RMSE',  numpy.min(numpy.sqrt(-scores))
    '''
    # print 'Coefficients', model.coef_


def random_forest(x, y):
    model = RandomForestRegressor(n_estimators=50,
                          max_features=len(x.columns),
                          max_depth= 9,
                          n_jobs=1)
    predicted = cross_validation.cross_val_predict(model, x, y, 10, 1, 0, None, 0)
    scores = cross_validation.cross_val_score(model, x, y,  cv=10, scoring='mean_squared_error')

    print 'All RMSEs',  numpy.sqrt(-scores)
    print 'Mean RMSE',  numpy.mean(numpy.sqrt(-scores))
    print 'Best RMSE',  numpy.min(numpy.sqrt(-scores))
    print 'Coefficients', model.feature_importances_


def svm(x, y):
    model = SVC()
    predicted = cross_validation.cross_val_predict(model, x, y, 10, 1, 0, None, 0)
    scores = cross_validation.cross_val_score(model, x, y,  cv=10, scoring='mean_squared_error')

    print 'All RMSEs',  numpy.sqrt(-scores)
    print 'Mean RMSE',  numpy.mean(numpy.sqrt(-scores))
    print 'Best RMSE',  numpy.min(numpy.sqrt(-scores))
    print 'Coefficients', model.feature_importances_


def neural_networks(x, y):
    model = neural_network.MLPRegressor([1,1,1,1,1], 'relu', 'adam', 0.0001, 200, 'constant', 0.001, 0.5, 200,
                                        True, None, 0.0001, False, False, 0.9, True, False, 0.1, 0.9, 0.999, 1e-08)
    predicted = cross_validation.cross_val_predict(model, x, y, 10, 1, 0, None, 0)
    scores = cross_validation.cross_val_score(model, x, y,  cv=10, scoring='mean_squared_error')

    print 'All RMSEs',  numpy.sqrt(-scores)
    print 'Mean RMSE',  numpy.mean(numpy.sqrt(-scores))
    print 'Best RMSE',  numpy.min(numpy.sqrt(-scores))
    print 'Coefficients', model.get_params(True)