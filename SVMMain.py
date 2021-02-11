import SVMHelperMethods
import pandas

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Import the wine quality train sample from the excel file
trainingSample = pandas.DataFrame(pandas.read_excel("Wine_Training.xlsx", header = 0))

# Import the wine quality test sample from the excel file
testSample = pandas.DataFrame(pandas.read_excel("Wine_Testing.xlsx", header = 0))

# Transform the training class set to a list of list object
qualityTrainingSample = trainingSample["quality"]

# Transform the training class set to a list of list object
trainingSampleList = trainingSample[["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]]

# Initialize the support vector classifier
supportVectorClassifier = SVC(kernel='linear')

# Fit the supportVectorClassifier using the training sample lists
supportVectorClassifier.fit(trainingSampleList, qualityTrainingSample)

# Predict the target property values of the test sample set
wineQualityPrediction = supportVectorClassifier.predict(testSample)

# Get the length of the first one third slice of the list
editedTrainingSampleLength = round(len(trainingSampleList) / 3)

# Predict the wine quality, without using the pH column
wineQualityPrediction = RemovePHColumn(trainingSampleList, qualityTrainingSample, testSample, supportVectorClassifier)
