from sklearn.cluster import KMeans
from sklearn import preprocessing
from copy import deepcopy
from sklearn.linear_model import LogisticRegression
from collections import Counter

"""
Returns the SVM prediction, after removing the pH column from the training and test samples

Attributes
    ----------
    trainingSample : DataFrame
        The DataFrame that contains the training samples for the SVM

    qualityTrainingSample : DataFrame
        The DataFrame that contains the target training value samples for the SVM
        
    testSample : DataFrame
        The DataFrame that contains the test value samples for the SVM
    
    supportVectorClassifier : SVM
        The support vector machine object
"""
def RemovePHColumn(trainingSample, qualityTrainingSample, testSample, supportVectorClassifier):

    # Get the training sample without the pH column
    removedPHTrainingSample = trainingSample[["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "sulphates", "alcohol"]]

    # Get the test sample without the pH column
    removedPHTestSample = testSample[["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "sulphates", "alcohol"]]
    
    # Fit the supportVectorClassifier using the training sample lists
    supportVectorClassifier.fit(removedPHTrainingSample, qualityTrainingSample)

    # Predict the target property values of the test sample set
    return supportVectorClassifier.predict(removedPHTestSample)
