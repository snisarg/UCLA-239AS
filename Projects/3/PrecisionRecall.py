from sklearn.cross_validation import KFold
import utility
import numpy
import math

K_VALUE = 100

test_error = []
test_index = -1
truePos = 0;
falsePos = 0;
falseNeg = 0;
precisionArray = []
recallArray = []

kf = KFold(100000, 10, True)
for train, test in kf:
    for threshold in [2, 2.5, 3, 3.5, 4, 4.5]:
        test_index += 1
        local_error = 0
    
        r, w, test_rows = utility.r_skiplist(test)
    
        u, v = utility.nmf(r, K_VALUE, w)
        uv = numpy.dot(u, v)
    
        # UV here is the 90% trained set. Comparison next
        for row in test_rows:
            ui = row[0]-1
            mi = row[1]-1
            
            if(uv[ui, mi] >= threshold):
                if(r[ui, mi] >= threshold):
                    truePos += 1
                else: falsePos += 1
                    
            else:
                if(r[ui, mi] >= threshold):
                    if(uv[ui, mi] < threshold):
                        falseNeg += 1
    
        test_error.append(local_error)
        print 'Precision for threshold: %f' %threshold
        pre = float (truePos) / (truePos + falsePos) 
        print pre
        precisionArray.append(pre)
        
        print 'Recall for threshold: %f' %threshold
        rec = float (truePos) / (truePos + falseNeg)
        print rec
        recallArray.append(pre)
    utility.plotROCForPR(precisionArray, recallArray)

print test_error
