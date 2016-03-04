import utility
import numpy


# Que 4 part A

r, w = utility.get_R()

print r


# Que 4 part B


r, w = utility.get_R()

for k in [10, 50, 100]:
    U, V = utility.nmf(r, k, w)
    print U.shape
    print V.shape
    uv = numpy.dot(U, V)
    lse = 0

    for lambda_val in [0.01, 0.1, 1]:
        lse = 0
        for i in range(len(r)):
            for j in range(len(r[i])):
                    #if w[i, j] == 1:
                    #lse += (((r[i, j] - uv[i, j])**2) + (lambda_val * (U[i, j]**2 + V[i, j]**2)))
                if i < 943 and j < 10:
                    u_term = U[i, j]**2
                else:
                    u_term = 0

                if i < 10 and j < 1682:
                    v_term = V[i, j]**2
                else:
                    v_term = 0

                lse += (w[i, j] * ((r[i, j] - uv[i, j])**2)) + (lambda_val * (u_term + v_term))
        print "for lambda"
        print lambda_val
        print 'For %d latent terms, LSE: %f' % (k, lse)


'''
Starting NMF decomposition with 10 latent features and 100 iterations.
('fit residual', 915.80409999999995)
('total residual', 245.42580000000001)
(943L, 10L)
(10L, 1682L)
for lambda
0.01
For 10 latent terms, LSE: 60695.495347
for lambda
0.1
For 10 latent terms, LSE: 64850.338755
for lambda
1
For 10 latent terms, LSE: 106398.772836
Starting NMF decomposition with 50 latent features and 100 iterations.
('fit residual', 605.60199999999998)
('total residual', 174.58510000000001)
(943L, 50L)
(50L, 1682L)
for lambda
0.01
For 50 latent terms, LSE: 30661.868826
for lambda
0.1
For 50 latent terms, LSE: 32299.068705
for lambda
1
For 50 latent terms, LSE: 48671.067495
Starting NMF decomposition with 100 latent features and 100 iterations.
('fit residual', 476.86939999999998)
('total residual', 132.8989)
(943L, 100L)
(100L, 1682L)
for lambda
0.01
For 100 latent terms, LSE: 17754.303971
for lambda
0.1
For 100 latent terms, LSE: 18583.902009
for lambda
1
For 100 latent terms, LSE: 26879.882383
'''