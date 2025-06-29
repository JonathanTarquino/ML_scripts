# Created by Jonathan Tarquino - jst104@case.edu - may 29,2025

from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis,QuadraticDiscriminantAnalysis
from corr_prunning import pick_best_uncorrelated_features
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.preprocessing import StandardScaler
import pandas as pd
from performance import performance_values
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from mrmr import mrmr_classif,mrmr_regression
from wilcoxon import wilcoxonFS
from sklearn.metrics import RocCurveDisplay, auc
from LASSO_selection import LassoFS
import math


def main_classifier(clfr,training_set ,
                    training_labels,testing_set,
                    testing_labels,options=None,
                    RF_max_depth=None,
                    RF_min_samples_split=2,
                    RF_n_estimator=80,
                    svm_kernel = 'linear'):

    print(np.shape(training_set))
    classifiers = {
    'LDA': LinearDiscriminantAnalysis(),
    'QDA': QuadraticDiscriminantAnalysis(),
    'SVM': svm.SVC(kernel = svm_kernel, C=1.0,probability=True),
    'RANDOMFOREST': RandomForestClassifier(n_estimators= RF_n_estimator, min_samples_split=RF_min_samples_split, max_depth = RF_max_depth)
    }

    clf = classifiers[clfr]
    print('Your selected classifier is:', clfr,clf)
    clf.fit(training_set, training_labels)
    # print(training_set)
    # print(testing_set)
    clf_prediction = clf.predict(testing_set)

    clf_scores = clf.predict_proba(testing_set)[:,1]

    return clf_prediction, clf_scores


def nFoldCV_withFS(data_set,data_labels,classifier='LDA',fsname='wilcoxon',
                   num_top_feats=1000000,feature_idxs=[1],
                   shuffle=1,nFolds=3,nIter=25,
                   classbalancetestfold='equal',
                   featnames = None,
                   full_fold_info =0, scaleFeatures = True,  with_corrPrun = True,
                   RF_max_depth=None,
                   RF_min_samples_split=2,
                   RF_n_estimator=80,
                   svm_kernel = 'linear',correlation_factor=0.6):
    # INPUTS:
                # data_set: Set of features as pandas DataFrame, used to train a model a validate it in Cross validation scheme
                # data_labels: numpy array conatinig binary labels
                # classifier: -- options: 'LDA','QDA','SVM','RANDOMFOREST' (Default: 'LDA');
                # classifieroptions: (Not working optional) A dditional {...,'Name','Value'...'} parameters for your classifier. Pass in as a cell array of strings
                # fsname -- options: 'mrmr','ttest','wilcoxon','lasso_fs_binarized') (Default: 'wilcoxon')
                # num_top_feats: -- the number of top features to select each iteration
                # feature_idxs: (optional) vector of pre-selected set of feature indices (i.e. from pruning or anti-correlation) (Default: include all variables in data_set)
                # shuffle: 1 for random, 0 for non-random partition (Default: 1)
                # nFolds: Number of folds to your cross-validation (Default: 3)
                # nIter: Number of cross-validation iterations (Default: 25)
                # subsets: (optional) training and testing subsets for each iteration (Default: computer will generate using 'nFold')
                # classbalancetestfold: (option) how to class balance the testing folds -- options: 'none' (default),'equal'
                # patient_ids: (needed for augmented data, i.e. multiple slices per patient) -- M x 1 vector of unique patient identifier (an integer) from which each data_set and data_label came from
                # osname: (optional) string for oversampling method) -- options: 'SMOTE','ADASYN','UNDERSAMPLE'
                # remouts: (optional) string for removing outlier method -- options: 'median','mean','quartile'
                # threshmeth: (not working option) string of optimal ROC threshold method -- options: 'euclidean', 'matlab' (Default: 'euclidean')
                # simplewhiten: (Not working optional) logical of whether to apply simplewhitening to the data -- options: true, false (Default: true)
                # fsprunecorr: (optional) logical of whether to applying correlation-based pruning per feature family to initial feature set before embedded feature selection -- options: true, false (Default: false)
                #featnames:(optional) cell array of feature names for each feature in "data_set" (REQUIRED if params.fsprune = true)
                # full_fold_info
                # scaleFeatures: Optional , to scale features (StandardScaler). By default se to True
                # with_corrPrun: True or False, given in order to avoid or include feature correlation filtering
    # %
    # % OUTPUTS:
    #           Single numpu array structure contatining multiple pandas Dataframes as follow:
    # %         iter_perf: data frame contatining all performance values
    #                       (12 columns namely 'tn', 'fp', 'fn', 'tp', 'auc_p', 'recall', 'Precision', 'Specificity', 'Sensibility', ...
    #                       'Accuracy', 'kappa', 'Fscore', 'matthew correlation coefficient-MCC'] ) along multiple iterations (each row for single iteration performance)
    #           full_estimates: Each row provide classifier scores (predict_proba), along all folds in a single iteration, full_it_labels,all_FS_ids, counts_FS/sum(counts_FS), names(testing_labels)[1]), where each row contains the confidence score for each observation in Testing set



    # ------------------------------------ Setting data sample space -------------------------------------------------------
    performances_stats = []
    performances_stats = pd.DataFrame(performances_stats)

    random_state = 12883823 # setting a seed to make the splitting process completely repeatible

    if scaleFeatures == True:
        scaler = StandardScaler()
        # print(np.shape(cleared_data))
        scaler.fit(data_set)


    # -------------------------------------- Data splitting -----------------------------------------------------------------------
    rkf = RepeatedStratifiedKFold(n_splits=nFolds, n_repeats=nIter, random_state=random_state)
    rkf.get_n_splits(data_set, data_labels)
    print(rkf, type(data_set))
    prev_iteration = 0
    all_FS = []
    full_predictions = [] # variable to save predictions along folds for a single iteration
    full_predictions = pd.DataFrame(full_predictions)
    full_labels = []
    full_labels = pd.DataFrame(full_labels)
    full_FS = []
    full_FS = pd.DataFrame(full_FS)
    full_pvals = []
    full_pvals = pd.DataFrame(full_pvals)
    full_estimates = []
    full_estimates = pd.DataFrame(full_estimates)
    iter_perf = []
    iter_perf = pd.DataFrame(iter_perf)
    pb_estimates_it = []
    pb_estimates_it =pd.DataFrame(pb_estimates_it)
    full_it_labels = []
    full_it_labels = pd.DataFrame(full_it_labels)
    fold_labels = []
    fold_labels = pd.DataFrame(fold_labels)
    all_predicted_labels = []
    all_predicted_labels = pd.DataFrame(all_predicted_labels)
    pivotF = []



    if with_corrPrun == True:
        num_features = math.ceil(0.5*np.shape(data_set)[1]) #what percent of the features: here=0.7
        idx = range(np.shape(data_set)[1])#[0,1,2,3] #but search through all available features
        correlation_metric = 'spearman'
        set_candiF, p_vals = pick_best_uncorrelated_features(data_set,data_labels, idx, num_features,correlation_factor,correlation_metric)

        feature_idxs = set_candiF # Features to keep after correlation prunning
        cleared_data = data_set.iloc[:,feature_idxs] # cleared_data corresponds to a version of features without correlation

    else:
        print('\n \t No Correlation prunning performed')
        cleared_data = data_set
        feature_idxs = np.arange(np.shape(data_set)[1])
        p_vals = np.zeros(np.shape(data_set)[1])


    for i, (train_index, test_index) in enumerate(rkf.split(data_set,data_labels)):

        print(f"Fold {i} from iteration {i//nFolds}:")

        # print(f"  Train: index={train_index}")
        # print(f"  Test:  index={test_index}")

        Train_data = cleared_data.iloc[train_index,:] #### This is previous and functional
        # Train_data = data_set.iloc[train_index,feature_idxs]


        # Train_data.columns.name = None
        Train_labels = data_labels[train_index]
        Test_data = cleared_data.iloc[test_index,:] #### This is previuos and functional
        # Test_data = data_set.iloc[test_index,feature_idxs]

        Test_labels = data_labels[test_index]
        # print(np.shape(Train_labels))


        ########################

        #---------------------- Feature selecction based on FSname ------------------------
        Fselections = {
            'mrmr': mrmr_regression,
            # 'ttest': t_testFS,
            'wilcoxon': wilcoxonFS,
            'lasso_fs_binarized':LassoFS
            }

        rankFeatures = Fselections[fsname](X=Train_data,y=Train_labels,K=num_top_feats)
        print('Selected feature selection method:',fsname)


        if fsname == 'mrmr':
            pivotF = rankFeatures
            for a in range(len(rankFeatures)):
                for b in range(np.shape(Train_data)[1]):

                    if rankFeatures[a] == Test_data.iloc[:,b].name:
                        rankFeatures[a] = b

        print('\n \t Retrieved Features:.............................................................', rankFeatures,feature_idxs, p_vals)
        print('\n \t',np.array(feature_idxs)[rankFeatures],'\n',np.array(p_vals)[rankFeatures])
        rankFeatures2pass = np.array(feature_idxs)[rankFeatures]
        pVals2pass = np.array(p_vals)[rankFeatures]

        # --------------------------------- Classification ------------------------------------------
        classifier_pred, classifier_score = main_classifier(classifier,Train_data.iloc[:,rankFeatures],Train_labels,Test_data.iloc[:,rankFeatures],Test_labels,RF_max_depth=None,
                    RF_min_samples_split=2,
                    RF_n_estimator=80,
                    svm_kernel = svm_kernel)


        if full_fold_info == 1:
            nFold_perf = performance_values(Test_labels,classifier_pred,classifier_score)
            performances_stats = pd.concat([performances_stats,pd.DataFrame(nFold_perf)],axis=1)
            pb_estimates_it = pd.concat([pb_estimates_it,pd.DataFrame(classifier_score)])

            # print(performances_stats)

        elif full_fold_info == 0 and prev_iteration == nFolds-1:
            prev_iteration =0
            # print(np.shape(full_predictions))
            classifier_pred= np.reshape(classifier_pred,(1,len(classifier_pred)))
            full_predictions = pd.concat([full_predictions,pd.DataFrame(classifier_pred)],axis=1)
            # print(np.shape(full_predictions))
            all_predicted_labels = pd.concat([all_predicted_labels,pd.DataFrame(full_predictions)],axis=0)
            full_labels = pd.concat([full_labels,pd.DataFrame(Test_labels)])
            print('\t End of iteration.............for iteration support :',np.shape(all_predicted_labels),'out of :',np.shape(full_labels))
            full_FS = pd.concat([full_FS,pd.DataFrame(rankFeatures2pass)])
            full_pvals = pd.concat([full_pvals,pd.DataFrame(pVals2pass)])
            pb_estimates_it = pd.concat([pb_estimates_it,pd.DataFrame(np.reshape(classifier_score,(1,len(classifier_score))))],axis=1)
            fold_labels =  pd.concat([fold_labels,pd.DataFrame(np.reshape(Test_labels,(1,len(Test_labels))))],axis=1)
            full_it_labels = pd.concat([full_it_labels,pd.DataFrame(fold_labels)],axis=0)
            full_estimates = pd.concat([full_estimates,pd.DataFrame(pb_estimates_it)],axis=0)
            all_FS = full_FS


            # full_FS.plot.hist(bins=12, alpha=0.5)
            # plt.show()

            # plt.hist(full_FS, bins = len(full_FS))
            # print('.........>',len(full_FS))
            # plt.show()


            full_labels = []
            full_labels = pd.DataFrame(full_labels)
            full_predictions = []
            full_predictions = pd.DataFrame(full_predictions)
            #### full_FS = []
            #### full_FS = pd.DataFrame(full_FS)
            pb_estimates_it = []
            pb_estimates_it =pd.DataFrame(pb_estimates_it)
            fold_labels = []
            fold_labels = pd.DataFrame(fold_labels)



            # print(np.mean(performances_stats.iloc[:,i-nFolds+1:i+1],axis=1),np.std(performances_stats.iloc[:,i-nFolds+1:i+1],axis=1))
        elif full_fold_info == 0:
            prev_iteration+=1
            full_labels = pd.concat([full_labels,pd.DataFrame(Test_labels)])
            classifier_pred= np.reshape(classifier_pred,(1,len(classifier_pred)))
            # print('full Labels\n',np.shape(full_labels),np.shape(classifier_pred))
            full_predictions = pd.concat([full_predictions,pd.DataFrame(classifier_pred)],axis=1)
            full_FS = pd.concat([full_FS,pd.DataFrame(rankFeatures2pass)])
            full_pvals = pd.concat([full_pvals,pd.DataFrame(pVals2pass)])
            pb_estimates_it = pd.concat([pb_estimates_it,pd.DataFrame(np.reshape(classifier_score,(1,len(classifier_score))))],axis=1)
            fold_labels =  pd.concat([fold_labels,pd.DataFrame(np.reshape(Test_labels,(1,len(Test_labels))))],axis=1)
            print('Moving iteration in progress with a support of..............', np.shape(full_predictions),np.shape(full_labels))


        # summary_stats = [np.mean(performances_stats,axis=1),np.std(performancconcates_stats,axis=1)]
        # performances_stats = pd.concat([performances_stats,pd.DataFrame(np.transpose(summary_stats))],axis=1)
        #
        # print(np.shape(full_it_labels))

    #  --------------------- Actual Classification performance computation ----------------------------
    print('\t Computing performance metrics\n',np.shape(full_it_labels),np.shape(all_predicted_labels),np.shape(full_estimates))


    fig, ax = plt.subplots(figsize=(6, 6))

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    for r in range(np.shape(full_estimates)[0]):
            nFold_perf,names = performance_values(full_it_labels.iloc[r,:],all_predicted_labels.iloc[r,:],full_estimates.iloc[r,:])
            iter_perf = pd.concat([iter_perf,pd.DataFrame(np.reshape(nFold_perf,(1,len(nFold_perf))))],axis=0)


            viz = RocCurveDisplay.from_predictions(
                full_it_labels.iloc[r,:],
                full_estimates.iloc[r,:],
                name=f"ROC iteration {r}",
                alpha=0.3,
                lw=1,
                ax=ax,
                plot_chance_level = False
                )
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            print('o.................................',viz.fpr)

            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)




    #
    mean_tpr = np.mean(tprs,axis = 0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(
        mean_fpr,
        mean_tpr,
        color="b",
        label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
        lw=2,
        alpha=0.8,
    )


    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(np.minimum(mean_tpr + std_tpr, 1),1)
    tprs_lower = np.maximum(np.maximum(mean_tpr - std_tpr, 0),0)
    ax.fill_between(
        mean_fpr,
        tprs_lower,
        tprs_upper,
        color="grey",
        alpha=0.2,
        label=r"$\pm$ 1 std. dev.",
    )

    ax.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title=f"Mean ROC curve with variability\n(Positive label ",
    )
    ax.legend(loc="lower right")
    print([all_FS,full_pvals])
    print([np.unique(all_FS, return_counts=True),' \n',np.unique(full_pvals, return_counts=True)])
    plt.show()
    #
    all_FS_ids, counts_FS = np.unique(all_FS, return_counts=True)

    print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx',featnames)

    if isinstance(featnames, pd.DataFrame):

        all_FS = featnames.iloc[all_FS_ids].to_numpy()
        # print(all_FS[:,1])
        # print(type(all_FS),np.shape(all_FS_ids))
        return(iter_perf,full_estimates, full_it_labels,all_FS[:,1], counts_FS/sum(counts_FS), names)

    else:

        print(type(all_FS_ids),np.shape(all_FS_ids))

        return(iter_perf,full_estimates, full_it_labels,all_FS_ids, counts_FS/sum(counts_FS), names)





