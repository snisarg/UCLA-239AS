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

lambda_values = [0.01]
r, w = utility.get_R()
k = 10
truePos = 0;
falsePos = 0;
falseNeg = 0;
precisionArray = []
recallArray = []
rows,cols = r.shape
threshold_ranges = [x/10.0 for x in range(0, 50, 5)]

# Run Regularized ALS for different values of lambda
for lambda_val in lambda_values:
    uv = utility.weightedRegALS(r, lambda_val, k, w, 20)
    for threshold in threshold_ranges:
        for i in range(len(r)):
            for j in range(len(r[i])):
                if(uv[i, j] >= threshold):
                    if(r[i, j] >= threshold):
                        truePos += 1
                    else: falsePos += 1
                        
                else:
                    if(r[i, j] >= threshold):
                        #if(uv[i, j] < threshold):
                            falseNeg += 1
        print 'Precision for threshold: %f' %threshold
        pre = float (truePos) / float (truePos + falsePos) 
        print pre
        precisionArray.append(pre)
        
        print 'Recall for threshold: %f' %threshold
        rec = float (truePos) / float (truePos + falseNeg)
        print rec
        recallArray.append(rec)

print 'Precision: '
print precisionArray
print 'Recall: '
print recallArray
utility.plotRocAlsPR(precisionArray, recallArray, len(threshold_ranges))

# Que 4 part C - ROC curve
