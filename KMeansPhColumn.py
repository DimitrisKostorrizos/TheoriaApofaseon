from sklearn.cluster import KMeans
from copy import deepcopy
from collections import Counter


"""
Returns the SVM prediction, after applying K-Means to the training samples and replacing the pH with average pH of the clustered samples

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
def KMeansPHColumn(trainingSample, qualityTrainingSample, testSample, supportVectorClassifier, editedTrainingSampleLength):

    # Deep copy part of the training sample to the logistic regression training sample
    kMeansPHTrainingSample = trainingSample[["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "sulphates", "alcohol"]][editedTrainingSampleLength:]

    # Deep copy part of the training sample to the logistic regression test sample
    kMeansPHTestSample = trainingSample[["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "sulphates", "alcohol"]][:editedTrainingSampleLength]

    # Initialise the K-Means clustering object
    kMeansClustering = KMeans()

    # Fit the K-Means clustering object
    kMeansClustering.fit(kMeansPHTrainingSample)

    # Get the training sample clustering labels
    trainingSampleClusteringLabels = kMeansClustering.labels_

    # Predict the clusters of the samples that miss the pH value
    testSampleClusteringPrediction = kMeansClustering.predict(kMeansPHTestSample)

    # Declare a dictionary for the clusters and the average pH values
    ClusterPHDictionary = {}

    # For every K-Means cluster
    for index in range(kMeansClustering.n_clusters):

        # Set the cluster's average pH value to 0
        ClusterPHDictionary[index] = 0

    # For every clustering label...
    for index in range(len(trainingSampleClusteringLabels)):

        # Sum the pH values of the training samples that belong in the same cluster
        ClusterPHDictionary[trainingSampleClusteringLabels[index]] += trainingSample.iloc[index, 8]

    # Initialise a Counter object to count the number of samples that belong to each cluster
    clusterNumberOfOccurrencesCounter = Counter(trainingSampleClusteringLabels)

    # For every cluster...
    for clusterEntry in ClusterPHDictionary.keys():

        # Get the average pH of the samples that belong to the cluster
        ClusterPHDictionary[clusterEntry] /= clusterNumberOfOccurrencesCounter[clusterEntry]

    # For every edited test sample in the first one third of the DataFrame...
    for index in range(len(trainingSample[:editedTrainingSampleLength])):

        # Update the pH value of the training sample
        trainingSample.iloc[index, 8] = ClusterPHDictionary.get(testSampleClusteringPrediction[index])

    # Fit the supportVectorClassifier using the training sample
    supportVectorClassifier.fit(trainingSample, qualityTrainingSample)

    # Predict the target property values of the test sample
    return supportVectorClassifier.predict(testSample)
