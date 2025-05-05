import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



def LDA(training_set , testing_set, training_labels, testing_labels, options=None):
    # print(training_set,testing_set)
    clf = LinearDiscriminantAnalysis()
    clf.fit(training_set, training_labels)
    LDA_prediction = clf.predict(testing_set)

    performance_values(testing_labels,LDA_prediction)
