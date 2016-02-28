from sklearn.cross_validation import KFold
import utility
import numpy
import math

K_VALUE = 100

test_error = []
test_index = -1

kf = KFold(100000, 10, True)
for train, test in kf:
    test_index += 1
    local_error = 0

    r, w, test_rows = utility.r_skiplist(test)
    # r, w = utility.get_R()

    # # Test for empty rows.
    # for row in w:
    #     sum = 0
    #     for element in row:
    #         sum += element
    #     if sum == 0:
    #         print sum
    #
    # print "Switching to columns now"
    # for column in w.T:
    #     sum = 0
    #     for element in column:
    #         sum += element
    #     if sum == 0:
    #         print sum

    u, v = utility.nmf(r, K_VALUE, w)
    uv = numpy.dot(u, v)

    # UV here is the 90% trained set. Comparison next
    for row in test_rows:
        ui = row[0]-1
        mi = row[1]-1
        local_error += math.sqrt((r[ui, mi] - uv[ui, mi])**2)

    test_error.append(local_error)

print test_error
