import utility
import numpy as np


# Que 4 part A
'''

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

lambda_values = [0.01, 0.1, 1]
r, w = utility.get_R()
k = 10

# Run Regularized ALS for different values of lambda
for lambda_val in lambda_values:
    utility.weightedRegALS(r, lambda_val, k, w, 20)


# Que 4 part C - ROC curve
