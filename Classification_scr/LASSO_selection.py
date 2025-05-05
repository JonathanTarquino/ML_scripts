import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso, LogisticRegression

def LassoFS(X,y,K=1000):
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
    labels = y

    if K > np.shape(data)[1]:
        K = np.shape(data)[1]


    f = np.shape(data)[1] # number of features
    p = np.zeros(f)
    s = np.zeros(f)


    idx2rem = data.isna()
    # print('idx2rem',idx2rem)
    idx2keep = [j for j in range(np.shape(idx2rem)[0]) if idx2rem.iloc[j,:].any() == False ]
    # print(idx2keep)
    tempdata = data.iloc[idx2keep,:]
    templabels = labels[idx2keep]
    # print('------',templabels)

    # print(pd.DataFrame(tempdata))
    sel_ = SelectFromModel(Lasso(alpha=0.001, random_state=10),max_features=K)
    sel_.fit(pd.DataFrame(tempdata),pd.DataFrame(templabels))
    Lasso_logic =sel_.get_support()
    SelectedF = [n for n in range(len(Lasso_logic)) if Lasso_logic[n] == True ]

    return SelectedF #,p,s
