from copy import deepcopy

"""
Returns the SVM prediction, after replacing one third of the pH column from the training samples, with the average pH of the rest of the training samples

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
def AveragePHColumn(trainingSample, qualityTrainingSample, testSample, supportVectorClassifier, editedTrainingSampleLength):

    # Deep copy the training sample list to the average pH training sample
    averagePHTrainingSample = deepcopy(trainingSample)

    # Initialise a sum to 0
    Average = 0

    # For every sample list in the non edited training sample list...
    for sample in averagePHTrainingSample["pH"][editedTrainingSampleLength:]:

        # Add the pH value to the sum
        Average += sample

    # Get the average pH in the non edited training sample list
    Average /= len(averagePHTrainingSample) - editedTrainingSampleLength

    # For every edited test sample in the first one third of the list...
    for sample in averagePHTrainingSample["pH"][:editedTrainingSampleLength]:

        # Remove the pH value
        sample = Average

    # Fit the supportVectorClassifier using the training sample lists
    supportVectorClassifier.fit(averagePHTrainingSample, qualityTrainingSample)

    # Predict the target property values of the test sample set
    return supportVectorClassifier.predict(testSample)
