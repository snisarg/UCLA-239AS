import utility
import numpy

r, w = utility.get_R()
predictedLiked = 0;
predictedLikedWrong = 0;
actuallyLiked = 0;
actuallyLikedPredictedWrong = 0
threshold = 2;

for k in [10, 50, 100]:
    for threshold in [2, 3, 4]:
        U, V = utility.nmf(r, k, w)
        uv = numpy.dot(U, V)
        for i in range(len(r)):
            for j in range(len(r[i])):
                if w[i, j] == 1:
                    if(uv[i, j] > threshold):
                        predictedLiked += 1
                        if(r[i, j] < threshold):
                            predictedLikedWrong += 1
                    
                    if(r[i, j] > threshold):
                        actuallyLiked += 1
                        if(uv[i, j] < threshold):
                            actuallyLikedPredictedWrong += 1
    
#         print 'predictedLiked %f' %predictedLiked
#         print 'predictedLikedWrong %f' %predictedLikedWrong
#         
        print 'Precision for threshold: %f' %threshold
        print float (predictedLiked - predictedLikedWrong) / (predictedLiked)
        
        print 'Recall for threshold: %f' %threshold
        print float (actuallyLiked - actuallyLikedPredictedWrong) / (actuallyLiked)