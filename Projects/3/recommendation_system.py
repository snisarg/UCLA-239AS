import utility
import numpy
import copy
from sklearn.cross_validation import KFold
import matplotlib.pyplot as pyplot

# Que 5 - Create a recommendation system
# Basic code structure. Need to modify as per question
# Run either nmf or Regularized ALS

# L - no of movies to be considered
def get_mean_precision(top_l_orig_matrix, top_l_predict_matrix):

    rows, cols = top_l_predict_matrix.shape
    precision = 0
    false_alarm = 0

    for i in range(rows):
        count = len(set(top_l_predict_matrix[i].flat).intersection(top_l_orig_matrix[i].flat))
        precision += count
        false_alarm += cols - precision

    mean_precision = (precision / (rows * cols))
    mean_false_alarm = (false_alarm / (rows * cols))
    return mean_precision, mean_false_alarm

# returns a matrix of top N values from each row of given matrix

def  gettopN(matrix, N, w):

    rows, cols = matrix.shape
    top_n_matrix = numpy.zeros((rows, N))

    for i in range(rows):
        weighted_list = numpy.multiply(w[i], matrix[i])
        top_n_list = numpy.argpartition(weighted_list.flat, -N)[-N:]
        top_n_matrix[i] = copy.copy(top_n_list)

    return top_n_matrix

# Que 5 part A

# find out top L movies from uv for each user (each row) & find average precision
# Uses 10-fold cross validation

#r, w = utility.get_R()
K_VALUE = 10
L = 5

kf = KFold(100000, 10, True)

for train, test in kf:
#    test_index += 1
#    local_error = 0

    r, w, test_rows = utility.r_skiplist(test)
    test_matrix = numpy.matrix(test_rows)
    # use Reg ALS matrix
    u, v = utility.nmf(w, K_VALUE, r)   # Swap R & W
    uv = numpy.dot(u, v)

    # Calculate mean precision across all folds with respect to Test matrix
    top_l_test_matrix = gettopN(test_matrix, L, w)
    top_l_predict_matrix = gettopN(uv, L, w)
    fold_precision = fold_precision + get_mean_precision(top_l_test_matrix,top_l_predict_matrix)

mean_fold_precision = fold_precision / 10

print "Que 5 part A output"
print "Mean Fold Precision"
print mean_fold_precision

# Que 5 part B
# hit-rate is assumed to be precision
# failure rate is total - precision

r = utility.get_R()
L = 5
hit_rate_list = []
false_alarm_list = []
recall_list = []
# replace this with RegALS( )
u, v = utility.nmf(w, K_VALUE, r)   # Swap R & W
uv = numpy.dot(u, v)

for i in range(1, L+1):
    top_l_orig_matrix = gettopN(r, L)
    top_l_predict_matrix = gettopN(uv, L)
    hit_rate, false_alarm = get_mean_precision(top_l_orig_matrix, top_l_predict_matrix)
    hit_rate_list.append(hit_rate)
    false_alarm_list.append(false_alarm)

# Plot false alarm rate vs hit-rate

pyplot.plot(false_alarm_list, hit_rate_list)
pyplot.show()
