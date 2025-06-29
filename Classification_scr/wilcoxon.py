from scipy import stats
import numpy as np

def wilcoxonFS(X,y,K=1000):

    # % INPUTS
    # %      data: a N*M matrix, indicating N samples, each having M dimensions.
    # %    labels: a N*1 matrix (vector), indicating the class/category of the N samples. Must be categorical.
    # %         K: (optional) the number of top features need to be returned
    # %  varargin: any additional parameters to wilcoxn
    # %
    # % OUTPUTS
    # %     fea: (if K specified): indices of the top K features based on rank (in descending order); otherwise, just sequential indices
    # %       p: p-values for significance testing for each feature
    # %       h: whether or not the null hypothesis was rejected (1) or failed to reject (0) for each feature
    #
    # % Code written by Jonathan Tarquino 04-2025

    data = X
    print(data)
    labels = y

    if K > np.shape(data)[1]:
        K = np.shape(data)[1]


    f = np.shape(data)[1] # number of features
    p = np.zeros(f)
    s = np.zeros(f)

    for i in range(f):
        idx2rem = data.iloc[:,i].isna()
        # print(data.iloc[:,i])
        idx2keep = [j for j in range(len(idx2rem)) if idx2rem.iloc[j] == False ]
        # print(idx2keep)
        tempdata = data.iloc[idx2keep,i]
        templabels = labels[idx2keep]
        # print('------',tempdata)
        resRank = stats.ranksums(tempdata.iloc[templabels==1], tempdata.iloc[templabels==-1])
        # print(i,'________________________',resRank)
        s[i] = resRank[0]
        p[i] = resRank[1]


    # print('Most relevant feature stats:',s,'\n','and correpoding c_values:',p)
    p_sorted = sorted(p)
    p_order = np.argsort(p)
    h_sorted = sorted(s)
    h_order = np.argsort(s)
    # print(':::::::::::',p_sorted,p_order,len(p_order),K)
    if len(p_order)>=1:
        if K==1:
            p_order_l = p_order
            print(p_order_l)
        else:
            p_order_l = p_order[0:K-1]
            print(p_order_l)

    print(':::::::::::',p_sorted,p_order_l)
    return p_order_l#, p_sorted
