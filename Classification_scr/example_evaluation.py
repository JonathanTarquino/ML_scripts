import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from corr_prunning import pick_best_uncorrelated_features
from CrossValidation import nFoldCV_withFS
from sklearn.metrics import RocCurveDisplay, auc
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from CrossValidation import main_classifier
from performance import performance_values
import sys

##This test run on IRIS database, but it can be modified to run any other features. If running from a folder different than classification  adding Folder_2/subfolder to the system path-- uncomment the next line and change the proper path
#sys.path.insert(0, '/home/amninder/Desktop/project/Folder_2/subfolder')

## load data from IRIS project .
# YOU WILL PROBABLY NEED pip install ucimlrepo, but only to get iris dataset

from ucimlrepo import fetch_ucirepo

# fetch dataset
iris = fetch_ucirepo(id=53)

# data (as pandas dataframes)
data = iris.data.features
y= iris.data.targets

# metadata
#print(iris.metadata)

# Just to visualize variable information. NOT in all the cases
print(iris.variables)
print(data)
print(y)

# Setting labels {1,-1}

data_labels = np.zeros(np.shape(y))
data_labels = np.where(y=='Iris-virginica',1,-1)
# print(data_labels)


# Set evaluation parameters
classifier='LDA';
fsname='mrmr';
num_top_feats=4;
shuffle = 1;
n = 3;
nIter = 25;
num_top_feats = 1;
subsets = {};
featnames = {'sepal length','spepal width','petal length','petal width'}
#
##### remove correlated features
# print(math.ceil(0.7*np.shape(data)[1]))

####
# num_features = math.ceil(0.5*np.shape(data)[1]) #what percent of the features: here=0.7
# idx = [0,1,2,3] #but search through all available features
# correlation_factor = 0.9999 # 0.999 is used in this example given  highly correlated features but it is set to 0.6 by default
# correlation_metric = 'spearman'
# set_candiF, p_vals = pick_best_uncorrelated_features(data,data_labels, idx, num_features,correlation_factor,correlation_metric)
# feature_idxs = set_candiF # pre-selected set of features
#
# print(feature_idxs)
# # clear num_features idx correlation_factor correlation_metric set_candiF
# cleared_data = data.iloc[:,feature_idxs]


# ----------------------------------------- DATA SPLITING (0.2 for testing, 0.8 for training) -----------------------------
X_train, X_test, y_train, y_test = train_test_split( data, data_labels, test_size=0.2, random_state=42)

# %% Cross validation and trainnig performance with remaining features using 0.8 data (training split)
stats = nFoldCV_withFS(X_train,y_train,classifier=classifier ,nFolds = 2,nIter = 4,full_fold_info = 0,fsname=fsname,num_top_feats = 3,with_corrPrun= False)
print('\n-------------------------------------------------------------------------------------------------------------')
print('----------------------------------> Obtained training results ------------------------------>\n Performance\n',
      stats[4],'\n',
      stats[0])

# Getting ordered selected features from the retrieved list of hits along the repeated k-fold
uniqueFS, countFS = np.unique(stats[3], return_counts=True)
# print('@@@@@@@@@@@@@',uniqueFS,countFS)
org_idx = np.argsort(countFS)
org_uniqueFS = uniqueFS[org_idx]
org_countFS = countFS[org_idx]
# print('***********************',org_uniqueFS,org_countFS)

# Using best features to perform a final evaluation

testing_results = main_classifier(classifier, X_train.iloc[:,org_uniqueFS], y_train, X_test.iloc[:,org_uniqueFS], y_test)
# print('>>>>>>>>>>>>>>>>>>>>>>',np.shape(testing_results[0]),np.shape(testing_results[1]))

st = performance_values(pd.DataFrame(y_test), pd.DataFrame(testing_results[0]), pd.DataFrame(testing_results[1]))

print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> OBTAINED TESTING PERFORMANCE: >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
print(stats[4])
print(np.reshape(st[0],(1,len(st[0]))))

# # -------------------- if you want to download performance results uncomment the next lines
# path_to_output = 'Home/Documents/project' # change path
# TrainPerf = pd.DataFrame(stats[0])
# perfNames = pd.DataFrame(stats[4])
# TestPerf = pd.DataFrame(np.reshape(st[0],(1,len(st[0]))))
# print(TrainPerf)
# print(TestPerf)
# TrainPerf.to_csv(path_to_output+'/training_performance.csv')
# perfNames.to_csv(path_to_output+'/performance_metric_names.csv')
# TestPerf.to_csv(path_to_output+'/testing_performance.csv')
