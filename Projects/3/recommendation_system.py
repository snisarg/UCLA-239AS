import utility
import numpy


# Que 5 - Create a recommendation system
# Basic code structure. Need to modify as per question
# Run either nmf or Regularized ALS

r, w = utility.get_R()
print r

k = 10
# W is passed instead of R. need to verify
U, V = utility.nmf(X=w, latent_features=k, mask=r)
uv = numpy.dot(U, V)

# find out top L movies from uv for each user (each row)