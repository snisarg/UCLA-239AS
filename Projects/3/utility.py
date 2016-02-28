import numpy
from numpy import dot
import scipy.linalg


def get_R():
    my_data = numpy.genfromtxt('../../Datasets/ml-100k/u.data', delimiter='\t')
    max_users = numpy.max(my_data[:, 0])
    max_movies = numpy.max(my_data[:, 1])
    r = numpy.zeros((max_users, max_movies))
    w = numpy.zeros((max_users, max_movies))
    for row in my_data:
        r[row[0]-1, row[1]-1] = row[2]
        w[row[0]-1, row[1]-1] = 1
    return r, w


def r_skiplist(skip_index):
    my_data = numpy.genfromtxt('../../Datasets/ml-100k/u.data', delimiter='\t')
    max_users = numpy.max(my_data[:, 0])
    max_movies = numpy.max(my_data[:, 1])
    r = numpy.zeros((max_users, max_movies))
    w = numpy.zeros((max_users, max_movies))
    test_rows = []
    i = 0
    for row in my_data:
        r[row[0]-1, row[1]-1] = row[2]
        if i not in skip_index:
            w[row[0]-1, row[1]-1] = 1
        else:
            test_rows.append(row)
        i += 1
    return r, w, test_rows


def nmf(X, latent_features, mask, max_iter=100, error_limit=1e-6, fit_error_limit=1e-6):
    """
    Decompose X to A*Y
    """
    eps = 1e-5
    print('Starting NMF decomposition with {} latent features and {} iterations.'.format(latent_features, max_iter))
    # X = X.toarray()  # I am passing in a scipy.linalg sparse matrix

    # mask COMMENTED OUT AS WE INTEND TO PASS THIS ON OUR OWN
    #mask = numpy.sign(X)

    # initial matrices. A is `random [0,1] and Y is A\X.
    rows, columns = X.shape
    A = numpy.random.rand(rows, latent_features)
    A = numpy.maximum(A, eps)

    Y = scipy.linalg.lstsq(A, X)[0]
    bool_mask = mask.astype(bool)
    # for i in range(columns):
    #     Y[:,i] = scipy.linalg.lstsq(A[bool_mask[:,i],:], X[bool_mask[:,i],i])[0]
    Y = numpy.maximum(Y, eps)

    masked_X = mask * X
    X_est_prev = dot(A, Y)
    for i in range(1, max_iter + 1):
        # ===== updates =====
        # Matlab: A=A.*(((W.*X)*Y')./((W.*(A*Y))*Y'));
        top = dot(masked_X, Y.T)
        bottom = (dot((mask * dot(A, Y)), Y.T)) + eps
        A *= top / bottom

        A = numpy.maximum(A, eps)
        # print 'A',  numpy.round(A, 2)

        # Matlab: Y=Y.*((A'*(W.*X))./(A'*(W.*(A*Y))));
        top = dot(A.T, masked_X)
        bottom = dot(A.T, mask * dot(A, Y)) + eps
        Y *= top / bottom
        Y = numpy.maximum(Y, eps)
        # print 'Y', numpy.round(Y, 2)

        # ==== evaluation ====
        if i == max_iter:
            # print('Iteration {}:'.format(i))
            X_est = dot(A, Y)
            err = mask * (X_est_prev - X_est)
            fit_residual = numpy.sqrt(numpy.sum(err ** 2))
            X_est_prev = X_est

            curRes = scipy.linalg.norm(mask * (X - X_est), ord='fro')
            print('fit residual', numpy.round(fit_residual, 4))
            print('total residual', numpy.round(curRes, 4))
            if curRes < error_limit or fit_residual < fit_error_limit:
                break

    return A, Y
