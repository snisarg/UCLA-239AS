import numpy
from numpy import dot
import scipy.linalg
import matplotlib.pyplot as plot
from nltk.metrics.scores import precision

def get_R():
    my_data = numpy.genfromtxt('../../Datasets/ml-100k/u.data', delimiter='\t')
    max_users = numpy.max(my_data[:, 0])
    max_movies = numpy.max(my_data[:, 1])
    r = numpy.zeros((max_users, max_movies))
    w = numpy.zeros((max_users, max_movies))
    for row in my_data:
        r[row[0]-1, row[1]-1] = row[2]
        w[row[0]-1, row[1]-1] = 1
    return r, w


def get_matrix_from_data(rows):
    # max_users = numpy.max(rows[:, 0])
    # max_movies = numpy.max(rows[:, 1])
    r = numpy.zeros((943, 1682))
    for row in rows:
        r[row[0]-1, row[1]-1] = row[2]
    return r


def r_skiplist(skip_index):
    my_data = numpy.genfromtxt('../../Datasets/ml-100k/u.data', delimiter='\t')
    max_users = numpy.max(my_data[:, 0])
    max_movies = numpy.max(my_data[:, 1])
    r = numpy.zeros((max_users, max_movies))
    w = numpy.zeros((max_users, max_movies))
    test_rows = []
    i = 0
    for row in my_data:
        r[row[0]-1, row[1]-1] = row[2]
        if i not in skip_index:
            w[row[0]-1, row[1]-1] = 1
        else:
            test_rows.append(row)
        i += 1
    return r, w, test_rows


def nmf(X, latent_features, mask, max_iter=100, error_limit=1e-6, fit_error_limit=1e-6):
    """
    Decompose X to A*Y
    """
    eps = 1e-5
    print('Starting NMF decomposition with {} latent features and {} iterations.'.format(latent_features, max_iter))
    # X = X.toarray()  # I am passing in a scipy.linalg sparse matrix

    # mask COMMENTED OUT AS WE INTEND TO PASS THIS ON OUR OWN
    #mask = numpy.sign(X)

    # initial matrices. A is `random [0,1] and Y is A\X.
    rows, columns = X.shape
    A = numpy.random.rand(rows, latent_features)
    A = numpy.maximum(A, eps)

    Y = scipy.linalg.lstsq(A, X)[0]
    bool_mask = mask.astype(bool)
    # for i in range(columns):
    #     Y[:,i] = scipy.linalg.lstsq(A[bool_mask[:,i],:], X[bool_mask[:,i],i])[0]
    Y = numpy.maximum(Y, eps)

    masked_X = mask * X
    X_est_prev = dot(A, Y)
    for i in range(1, max_iter + 1):
        # ===== updates =====
        # Matlab: A=A.*(((W.*X)*Y')./((W.*(A*Y))*Y'));
        top = dot(masked_X, Y.T)
        bottom = (dot((mask * dot(A, Y)), Y.T)) + eps
        A *= top / bottom

        A = numpy.maximum(A, eps)
        # print 'A',  numpy.round(A, 2)

        # Matlab: Y=Y.*((A'*(W.*X))./(A'*(W.*(A*Y))));
        top = dot(A.T, masked_X)
        bottom = dot(A.T, mask * dot(A, Y)) + eps
        Y *= top / bottom
        Y = numpy.maximum(Y, eps)
        # print 'Y', numpy.round(Y, 2)

        # ==== evaluation ====
        if i == max_iter:
            # print('Iteration {}:'.format(i))
            X_est = dot(A, Y)
            err = mask * (X_est_prev - X_est)
            fit_residual = numpy.sqrt(numpy.sum(err ** 2))
            X_est_prev = X_est

            curRes = scipy.linalg.norm(mask * (X - X_est), ord='fro')
            print('fit residual', numpy.round(fit_residual, 4))
            print('total residual', numpy.round(curRes, 4))
            if curRes < error_limit or fit_residual < fit_error_limit:
                break

    return A, Y

# Weighted Regularized Alternating Least Squares (ALS)

def weightedRegALS(Q, lambda_, n_factors, W, n_iterations):
    m, n = Q.shape

    X = 5 * numpy.random.rand(m, n_factors)
    Y = numpy.linalg.lstsq(X, Q)[0]

    weighted_errors = []
    totalError =0
    for ii in range(n_iterations):
        for u, Wu in enumerate(W):
            X[u] = numpy.linalg.solve(numpy.dot(Y, numpy.dot(numpy.diag(Wu), Y.T)) + lambda_ * numpy.eye(n_factors),
                                   numpy.dot(Y, numpy.dot(numpy.diag(Wu), Q[u].T))).T
        for i, Wi in enumerate(W.T):
            Y[:,i] = numpy.linalg.solve(numpy.dot(X.T, numpy.dot(numpy.diag(Wi), X)) + lambda_ * numpy.eye(n_factors),
                                     numpy.dot(X.T, numpy.dot(numpy.diag(Wi), Q[:, i])))
        if(ii == n_iterations - 1):
            print('Total Error {}'.format(get_error(numpy.dot(X, Y), Q, W)))
            print('Absolute Error {}'.format(get_abs_error(numpy.dot(X, Y), Q, W)))
    weighted_Q_hat = numpy.dot(X,Y)
    return weighted_Q_hat

def get_error(R_hat, R, W):
    return numpy.sum((W * (R_hat - R))**2)

def get_abs_error(R_hat, R, W):
    tmp = W *numpy.abs(R_hat - R)
    return numpy.mean(tmp[W > 0.0])


def plotROCForPR(precisionArray, recallArray, threshold_size):
    length = len(precisionArray)
    #precision_avg = [1.0, 1.0, 1.0, 0.9879690921446729, 0.9826979770255159, 0.9669021748518717, 0.959645789925236, 0.9420445702157765, 0.9348069730072782, 0.9242778151681909, 0.9331056110612402, 0.9400793650793651, 0.9457194210576104, 0.945701279144237, 0.9463313207245392, 0.9412670298709355, 0.9392994564870782, 0.9311227100603827, 0.92774290802612, 0.9224143048419331, 0.9272150136437044, 0.9314482525510948, 0.9352018641238783, 0.9360550190225344, 0.9371875560158537, 0.9347560568326033, 0.9340148313384614, 0.9288106788477707, 0.9265841049519324, 0.9232353935965071, 0.9264654523115246, 0.9294289625540609, 0.9321525906715635, 0.9328463986656045, 0.9337486907212771, 0.9320600502169414, 0.9314866913401527, 0.9276137342797081, 0.925961535273699, 0.9233442011188654, 0.9257905873116681, 0.928080761843493, 0.9302220901864405, 0.9307343407490323, 0.9314291711817001, 0.9302089937348944, 0.9297998507361422, 0.9269329438421062, 0.9257372981112867, 0.9236800574266728, 0.9256424233385958, 0.9275038605779837, 0.9292686685075711, 0.9297517799097685, 0.9303661632379489, 0.9292920277018707, 0.9288670722402652, 0.9265379653929985, 0.9254544170172937, 0.9236217702034085, 0.9252651963421195, 0.9268363069534261, 0.9283379893455036, 0.9286396046251097, 0.9291072537492883, 0.9282114984847835, 0.9279047238282692, 0.9258225035761025, 0.9248543880138941, 0.9233006343896448, 0.9247186192004496, 0.9260831129320063, 0.9273943634347174, 0.9277554746763909, 0.9282424643556438, 0.9274654069574934, 0.9271647667629195, 0.9253247238787742, 0.9244943241050821, 0.9231996019526726, 0.9244444156294623, 0.9256476225263061, 0.9268105804753898, 0.9270949137604465, 0.9274862783461484, 0.9268625339845535, 0.9266584867075665, 0.9251208691662277, 0.924503097384105, 0.9234188124307139, 0.9245241081493006, 0.9255957293024262, 0.9266333322109611, 0.9269404181001436, 0.9273593526540882, 0.926749326727728, 0.9265203775163636, 0.9250382153434604, 0.9243925248458527, 0.9233555960865348]
    #recall_avg = [1.0, 0.9991, 0.9974333333333333, 0.9960667884693464, 0.9901261881350377, 0.9806061667834618, 0.9585883937316356, 0.9392272207880147, 0.9057830694753369, 0.8914557002303766, 0.9037113145979879, 0.9133076993314192, 0.9208322971068557, 0.9266071383143809, 0.9296628878047247, 0.9296712381346048, 0.9226277067425819, 0.9149419738932802, 0.8994147887096253, 0.8923728759483656, 0.8988285537767594, 0.9044404330087088, 0.9092369736017609, 0.9132653841643424, 0.9157523949459222, 0.9163458975066532, 0.9125319140313669, 0.9075880187328114, 0.8978482863919233, 0.893273493049092, 0.8976244735887285, 0.901556251052712, 0.9050446110526852, 0.9080421273358065, 0.9100211553421639, 0.9104187118820314, 0.9075742908562285, 0.9038788864822129, 0.8963915821771183, 0.892959048666571, 0.8962662011647846, 0.8993091873585782, 0.9020121339094758, 0.904395165516206, 0.9059788261976829, 0.9064343734909531, 0.9041387263711527, 0.9012776295186521, 0.8953463028633994, 0.8926564767513333, 0.8953261358860147, 0.8978322538687219, 0.9001414400348743, 0.9021549956084346, 0.903497372604784, 0.9039388414334151, 0.9021739843263272, 0.899756589050179, 0.8947350448510407, 0.89242089115243, 0.8946590963468144, 0.8967653047290426, 0.8987234569528751, 0.9004578476259572, 0.9017139028114476, 0.90219243461268, 0.9009299539222109, 0.898993537137907, 0.8947179821891186, 0.8926881681266049, 0.8946075012072758, 0.8964330270070813, 0.8981333425428675, 0.8996547142046362, 0.9007469651405118, 0.9010944413465884, 0.8998880248428605, 0.8982439274601871, 0.8946017931140329, 0.8928255036551245, 0.8945063769188786, 0.8961106341451363, 0.8976326466062055, 0.8989711804587258, 0.8999473640632577, 0.9003387384439651, 0.8992688214255817, 0.8977781977601577, 0.8945499481556184, 0.8929198038463226, 0.894414959346869, 0.8958400178467902, 0.8971733853495268, 0.8983598246223465, 0.899226623540896, 0.8994708162845512, 0.8984810129863733, 0.8971553807625289, 0.8942448865371567, 0.8928171402867843]
    precision_avg = []
    recall_avg = []
    plot.figure()
    for i in range(threshold_size):
        recall_avg.append(numpy.mean([recallArray[x] for x in range(i, length, threshold_size)]))
        precision_avg.append(numpy.mean([precisionArray[x] for x in range(i, length, threshold_size)]))
    plot.plot(recall_avg, precision_avg, label="ROC curve", linewidth=2)
    plot.plot([0, 1], [0, 1], 'k--')
    plot.ylim([0.4, 1.0])
    plot.xlim([0.4, 1.0])
    plot.ylabel('Recall')
    plot.xlabel('Precision')
    plot.show()


def plotRocAlsPR(precisionArray, recallArray, threshold_size):
    plot.figure()
    plot.plot(recallArray, precisionArray, label="ROC curve", linewidth=2)
    plot.plot([0, 1], [0, 1], 'k--')
    plot.ylim([0.0, 1.0])
    plot.xlim([0.0, 1.0])
    plot.ylabel('Precision')
    plot.xlabel('Recall')
    plot.show()


#plotROCForPR([], [], 5)
