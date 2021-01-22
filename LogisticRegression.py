from sklearn import preprocessing
from copy import deepcopy
from sklearn.linear_model import LogisticRegression


"""
Returns the SVM prediction, after applying Logistic Regression to the pH column of the training samples

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
        
    editedTrainingSampleLength : int
        The length of the one third of the training samples
"""
def LogisticRegressionPHColumn(trainingSample, qualityTrainingSample, testSample, supportVectorClassifier, editedTrainingSampleLength):

    # Deep copy  the training sample list to the average pH training sample list
    logisticRegressionPHTrainingSample = deepcopy(trainingSample)

    # Initialise a logistic regression classifier object
    logisticRegressionClassifier = LogisticRegression(solver="liblinear")

    # Deep copy part of the training sample list to the logistic regression training sample list
    logisticRegressionTrainingSample = trainingSample[["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "sulphates", "alcohol"]][editedTrainingSampleLength:]

    # Deep copy part of the training sample list to the logistic regression test sample list
    logisticRegressionTestSample = trainingSample[["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "sulphates", "alcohol"]][:editedTrainingSampleLength]

    # Declare a training target sample list for the logistic regression
    logisticRegressionTrainingTargetSampleList = trainingSample["pH"][editedTrainingSampleLength:]

    # Initialise a label encoder
    labelEncoder = preprocessing.LabelEncoder()

    # Transform the continuous float values to multi-class int values
    transformedTargetSamplesValues = labelEncoder.fit_transform(logisticRegressionTrainingTargetSampleList)

    # Fit the logisticRegression using the training sample
    logisticRegressionClassifier.fit(logisticRegressionTrainingSample, transformedTargetSamplesValues)

    # Predict the target property values of the test sample
    winePHPrediction = logisticRegressionClassifier.predict(logisticRegressionTestSample)

    # Inverse transform the multi-class int values to continuous float values
    winePHPrediction = labelEncoder.inverse_transform(winePHPrediction)

    # For every training sample...
    for index in range(len(logisticRegressionPHTrainingSample["pH"][:editedTrainingSampleLength])):

        # Remove the pH values
        logisticRegressionPHTrainingSample[index] = winePHPrediction[index]

    # Fit the supportVectorClassifier using the training sample lists
    supportVectorClassifier.fit(trainingSample, qualityTrainingSample)

    # Predict the target property values of the test sample
    return supportVectorClassifier.predict(testSample)
