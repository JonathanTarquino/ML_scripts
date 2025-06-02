from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_fscore_support,precision_score, recall_score
import numpy as np
import pandas as pd

def performance_values(testing_labels,predictions,scores,targetclass_name=None):
    # This function provide some performance metrics for a given unknown model (unknown for this function), including AUC, recall, precision, specificity, sensibility, accuracy, kappa, and F1-score, matthew correlation coefficient-MCC based on reference labels and predictions
    # INPUTS: testing_labels: pandas series like list containing original reference labels
    #         predictions: pandas series or DataFrame containing predicted labels
    #         scores: pandas series containing estimated model outcome probabilities in the way 'predict_proba' build in function provides for sklearn

    # print((testing_labels),predictions,(scores))
    # print('pppppppppppppppp',type(testing_labels),predictions,scores)
    testing_labels = testing_labels.iloc[:]
    predictions = predictions.iloc[:]
    scores =scores.iloc[:]
    # print(np.shape(testing_labels),np.shape(predictions))
    if targetclass_name == None:
        targetclass_name = 1
    stats_names = ['tn', 'fp', 'fn', 'tp','auc_p','recall','Precision', 'Specificity', 'Sensibility','Accuracy','kappa','Fscore','matthew correlation coefficient-MCC']

    stats = np.zeros(13)
    tn, fp, fn, tp = confusion_matrix(testing_labels,predictions).ravel()
    stats[0:4]= [tn, fp, fn, tp]
    fpr, tpr, thresholds = roc_curve(testing_labels, scores, pos_label=targetclass_name)
    # print('ROC thresholds to fpr & tpr +++++++++++++++++++++++++++',thresholds)
    auc_p = auc(fpr, tpr)
    stats[4] = auc_p
    #precision = precision_score(y_true, y_predprecision_score(y_true, y_pred, average='macro')
    recall = recall_score(testing_labels, predictions, average='macro')
    stats[5] = recall
    Pre = (tp)/(tp+fp)
    stats[6] = Pre
    spec = tn/(tn+fp)
    stats[7] = spec
    sens = tp/(tp+fn)
    stats[8] = sens
    acc = (tp+tn)/(tp+tn+fp+fn)
    stats[9] = acc
    kappa = (acc - Pre) / (1 - Pre)
    stats[10] = kappa
    Fscore = 2*tp/(2*tp+fp+fn)
    stats[11] = Fscore
    MCC = np.power((tp*tn - fp*fn) / ((tp+fp)*(fn+tn)*(fp+tn)*(tp+fn)),0.5) #%matthew correlation coefficient:
    stats[12] = MCC

    return stats, stats_names
