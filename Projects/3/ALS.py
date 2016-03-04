import utility
import numpy as np


# Que 4 part A


r, w = utility.get_R()

for k in [10, 50, 100]:
    U, V = utility.nmf(X=w, latent_features=k, mask=r)
    uv = np.dot(U, V)
    lse = np.sum((r * (w - uv)**2))
    #lse = 0
    #lse = (r - uv) ** 2
    #lse = sum(sum(np.multiply(w,lse)))

    print 'For %d latent terms, LSE: %f' % (k, lse)

'''
correct output Que 4 part A

Starting NMF decomposition with 10 latent features and 100 iterations.
('fit residual', 935.46429999999998)
('total residual', 17.4328)
For 10 latent terms, LSE: 82.043269
Starting NMF decomposition with 50 latent features and 100 iterations.
('fit residual', 571.5018)
('total residual', 26.6343)
For 50 latent terms, LSE: 192.148042
Starting NMF decomposition with 100 latent features and 100 iterations.
('fit residual', 378.60300000000001)
('total residual', 23.007999999999999)
For 100 latent terms, LSE: 144.922050
'''


# Que 4 part B

def weightedRegALS(Q, lambda_, n_factors, W, n_iterations):
    m, n = Q.shape

    X = 5 * np.random.rand(m, n_factors)
    Y = np.linalg.lstsq(X, Q)[0]

    weighted_errors = []
    totalError =0
    for ii in range(n_iterations):
        for u, Wu in enumerate(W):
            X[u] = np.linalg.solve(np.dot(Y, np.dot(np.diag(Wu), Y.T)) + lambda_ * np.eye(n_factors),
                                   np.dot(Y, np.dot(np.diag(Wu), Q[u].T))).T
        for i, Wi in enumerate(W.T):
            Y[:,i] = np.linalg.solve(np.dot(X.T, np.dot(np.diag(Wi), X)) + lambda_ * np.eye(n_factors),
                                     np.dot(X.T, np.dot(np.diag(Wi), Q[:, i])))
        # weighted_errors.append(get_error(Q, X, Y, W))
        # totalError += get_error(Q, X, Y, W)
        if(ii == n_iterations - 1):
            print('Total Error {}'.format(totalError))
    weighted_Q_hat = np.dot(X,Y)
    return weighted_Q_hat

lambda_values = [0.01, 0.1, 1]
r,w = utility.get_R()
k = 10

for lambda_val in lambda_values:
    weightedRegALS(r,lambda_val, k, w, 20)


