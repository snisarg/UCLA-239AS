import utility
import numpy

r, w = utility.get_R()

for k in [10, 50, 100]:
    U, V = utility.nmf(r, k, w)
    uv = numpy.dot(U, V)
    lse = 0

    for i in range(len(r)):
        for j in range(len(r[i])):
            if w[i, j] == 1:
                lse += (r[i, j] - uv[i, j])**2

    print 'For %d latent terms, LSE: %f' % (k, lse)

