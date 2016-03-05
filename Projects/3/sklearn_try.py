import utility
import numpy
from sklearn.decomposition import NMF

r, w = utility.get_R()

for k in [10, 50, 100]:
    model = NMF(n_components=k, init='random', random_state=0)
    W = model.fit_transform(r)
    H = model.components_

    print W.shape
    print H.shape

    uv = numpy.dot(W, H)
    lse = 0

    for i in range(len(r)):
        for j in range(len(r[i])):
            if w[i, j] == 1:
                lse += (r[i, j] - uv[i, j])**2

    print 'For %d latent terms, LSE: %f' % (k, lse)