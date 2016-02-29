import utility
import numpy
from nltk.metrics.scores import precision

r, w = utility.get_R()
predictedLiked = 0;
predictedLikedRight = 0;
actuallyLiked = 0;
actuallyLikedPredicted = 0
threshold = 2;
precisionArray = []
recallArray = []

for k in [10]:
    for threshold in [2, 2.5, 3, 3.5, 4, 4.5]:
        U, V = utility.nmf(r, k, w)
        uv = numpy.dot(U, V)
        for i in range(len(r)):
            for j in range(len(r[i])):
                if w[i, j] == 1:
                    if(uv[i, j] > threshold):
                        predictedLiked += 1
                        if(r[i, j] > threshold):
                            predictedLikedRight += 1
                    
                    if(r[i, j] > threshold):
                        actuallyLiked += 1
                        if(uv[i, j] > threshold):
                            actuallyLikedPredicted += 1
    
#         print 'predictedLiked %f' %predictedLiked
#         print 'predictedLikedWrong %f' %predictedLikedWrong
#         
        print 'Precision for threshold: %f' %threshold
        pre = float (predictedLikedRight) / (predictedLiked) 
        print pre
        precisionArray.append(pre)
        
        print 'Recall for threshold: %f' %threshold
        rec = float (actuallyLikedPredicted) / (actuallyLiked)
        print rec
        recallArray.append(pre)
        
    utility.plotROCForPR(precisionArray, recallArray)