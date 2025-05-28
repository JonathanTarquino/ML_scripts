import numpy as np
import pandas as pd
import math
from scipy import stats

def pick_best_uncorrelated_features(data,classes,idx_pool,num_features=100000000,correlation_factor=0.6,correlation_metric='spearman'):

# % Eliminate correlated features from large feature set giving priority to
# % significant features based on ranksum test
# %
# % Compare with: rankfeatures() in MATLAB using the CCWeighting parameter
# %
# % INPUTS:
# %   data = MxN feature matrix with M observations and N variables
# %   classes = Mx1 vector of class labels corresponding to observations in data
# %   num_features = maximum size of uncorrelated feature set to be returned ([DEFAULT: returns entire set of uncorrelated features]
# %   idx_pool = vector of indices to include in feature pool (i.e. for per-class feature selection). [DEFAULT: all features]
# %   correlation_factor = minimum correlation between features to trim [DEFAULT: 0.6]
# %   correlation_metric = 'pearson' (parametric) or 'spearman' (nonparametric) [Default: 'spearman']
# %
# % OUTPUTS:
# %   set_candiF = vector of sortd feature indices of those features which were uncorrelated yet most discriminative


    # CHECK INPUTS
    # print(data,classes)
    if int(np.unique(classes)[0]) != -1 or int(np.unique(classes)[0]) != -1:
            print('Error: unable to convert. Class labels have to be a vector [1,-1]. provided',np.unique(classes))
            return []

    elif np.shape(data)[0]!=len(classes):
        print('Error: CLASSES must contain same number of labels as observations in DATA')

    # %% DEFAULTS

    if len(idx_pool) > np.shape(data)[1]:
        idx_pool = range(np.shape(data)[2])

    if num_features > np.shape(data)[1]:
        num_features = len(idx_pool)
        warning_numfeatures = False
    elif num_features == len(idx_pool):
        warning_numfeatures = False
    else:
        warning_numfeatures = True

    if correlation_factor>1:
        print('Warning: provided correlation factor is higher than 1...Setting correlation_factor to 0.6')
        correlation_factor=0.6;

    correlation_metric = correlation_metric.lower()
    if correlation_metric not in ['spearman','pearson']:
        print('Warning: provided correlation_metric is not in list [spearman or pearson],... Setting to spearman ')
        correlation_metric='spearman';

    # PRELIM CHECK: GET RID OF BAD FEATURES

    pd_data = pd.DataFrame(data)
    # print(data,pd_data)
    # Remove any features (as columns) with nans (non-optional)
    nan_detect= pd.notna(pd_data).all(axis = 0)
    # print(nan_detect)
    idxs2keep = [x for x in range(len(nan_detect)) if nan_detect[x]==True]


    #
    # 'data' must not contain too few unique values (#unique must be >= 10% of observations)
    for i in range(np.shape(pd_data)[1]):
        if len(pd.unique(pd_data.iloc[:,i])) <= math.floor(0.1*np.shape(data)[0]):
            idxs2keep = [w for w in idxs2keep if w != i]


    # print('----------------------------',idxs2keep)
    #pd_data = pd_data.dropna(axis=1)
    idx_agree = [t for t in idx_pool if t in idxs2keep ]
    print('__________',idx_agree)
    if len(idx_agree)<1:
        print('Error: no idx_pool features retained after quality check')
        return []
    print(np.shape(pd_data))

    # %% KEEP DATA ORGANIZED BY P-VALS SO THAT WE GIVE PREFERENCE TOWARDS MORE DISCRIMINATIVE FEATURES
    #
    pos = np.where(classes==1)
    neg = np.where(classes==-1)
    p_values = []
    for j in idx_agree:
        if correlation_metric == 'pearson':
            statistic,p = stats.ttest_ind(pd_data.iloc[classes==1,j], pd_data.iloc[classes==-1,j])
        elif correlation_metric == 'spearman':
            # print('.................',len(classes==1),len(pd_data.iloc[:,j]))
            statistic,p = stats.ranksums(pd_data.iloc[classes==1,j], pd_data.iloc[classes==-1,j])
        p_values.append(p)
    print('p values:',p_values)
    keepIdx = np.argmin(p_values)


    copy_pvals = p_values
    copy_idx_agree = idx_agree

    set_mostdisF = [idx_agree[keepIdx]] # keep the most discriminative feature according to statistical test
    # print('8888888888',set_mostdisF,correlation_factor)
    RHO = pd_data.corr(method = correlation_metric) #np.corrcoef(pd_data[:,idx_pool],'Type',correlation_metric) #how correlated are the rest of the features with this feature?
    print(keepIdx, type(RHO))
    print(RHO.iloc[:,keepIdx])
    correlated = np.where(np.abs(RHO.iloc[keepIdx,:])>correlation_factor)[0] #identify the features which are correlated
    print('===================================================================================================================================',correlated)
    idx_agree = np.delete(idx_agree, correlated-1)  # remove these features from our pool
    # print(p_values)
    p_values = np.delete(p_values, correlated-1)
    # print(idx_agree,idx_pool,'\n',p_values)
    # print('------------<',set_mostdisF)
    # ITERATE

    while len(set_mostdisF)<num_features and len(idx_agree)>1: #and now repeat this scheme for the rest of the feature pool...

        keepIdx = np.argmin(p_values)
        print('init',keepIdx,set_mostdisF,idx_agree)
        set_mostdisF = list(set_mostdisF)
        set_mostdisF.append(idx_agree[keepIdx])  # keep the next most discriminative feature
        RHO = np.corrcoef(pd_data.iloc[:, idx_agree], rowvar=False)
        if np.shape(RHO):
            print(RHO)
            correlated = np.where(np.abs(RHO[keepIdx, :]) > correlation_factor)[0]
            print(correlated)
            idx_agree = np.delete(idx_agree, correlated)  # remove these features from our pool [NOTE: This will also remove the current feature of interest]
            p_values = np.delete(p_values, correlated)
            print(idx_agree,p_values)
        else:
            print('======',RHO)
            correlated = np.where(np.abs(RHO) > correlation_factor)[0]
            print(correlated)
            idx_agree = np.delete(idx_agree, correlated)  # remove these features from our pool [NOTE: This will also remove the current feature of interest]
            p_values = np.delete(p_values, correlated)
            print(idx_agree,p_values)




        # set_mostdisF = list(set_mostdisF)
        # print(set_mostdisF,keepIdx)
        # set_mostdisF.append(idx_agree[keepIdx])  # keep the next most discriminative feature
        # print('00000000000',idx_agree,set_mostdisF,keepIdx)
        # RHO = np.corrcoef(pd_data.iloc[:, idx_agree], rowvar=False)  # how correlated are the rest of the features with this feature?
        # correlated = np.where(np.abs(RHO[:,keepIdx]) > correlation_factor)[0]  # identify the features which are correlated
        # print('*****',correlated)
        # idx_agree = np.delete(idx_agree, correlated)  # remove these features from our pool [NOTE: This will also remove the current feature of interest]
        # idx_agree = np.delete(idx_agree,keepIdx)
        # p_values = np.delete(p_values, correlated)
        # p_values = np.delete(p_values, keepIdx)
        # print(idx_agree,p_values)

    if len(set_mostdisF)<num_features:
      print(f'Too many correlated features. Only able to return {len(np.unique(set_mostdisF))} num_features.')

    # print('XXXXXXXXXXX',set_mostdisF)
    # set_mostdisF = sorted(np.unique(set_mostdisF))
    # copy_pvals = [copy_pvals[t] for t in set_mostdisF]


    # Create a mapping of original indices to p-values
    pvals_dict = dict(zip(copy_idx_agree,copy_pvals))

    # When creating final result, use the mapping to get correct p-values
    set_mostdisF = sorted(np.unique(set_mostdisF))
    copy_pvals = [pvals_dict[idx] for idx in set_mostdisF if idx in pvals_dict]

    print(set_mostdisF,copy_pvals)

    return set_mostdisF,copy_pvals
