# Importing the dependencies
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Data Collection and Analysis

diabetes_dataset = pd.read_csv('/content/diabetes.csv')
diabetes_dataset.head()

# Printing the dimension of the dataset
diabetes_dataset.shape

# Getting the statistical measures of the dataset
diabetes_dataset.describe()

diabetes_dataset['Outcome'].value_counts()

# 0 ---> Non Diabetic
# 1 ---> Diabetic

diabetes_dataset.groupby('Outcome').mean()

# Separating the data lables
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']

print(X)
print(Y)

# Data Standardization
scalar = StandardScaler()

scalar.fit(X)

standardized_data = scalar.transform(X)

print(standardized_data)

X = standardized_data

print(X)
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

# Training the model

classifier = svm.SVC(kernel='linear')

# Training the support vector machine classifier

# Model Evaluation
# Accuracy Score

X_train_predict = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_predict, Y_train)

print("Accuracy Score of the training data: ", training_data_accuracy)

# For testing data
X_test_predict = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_predict, Y_test)

print("Accuracy score of the testing data: ", test_data_accuracy)

# Making the predictive System

input_data = (2,197,70,45,543,30.5,0.158,53)

# changing the input data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for only one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

# standardized the input data
std_data = scalar.transform(input_data_reshaped)
print(std_data)

prediction = classifier.predict(std_data)
print(prediction)

if(prediction[0] == 0):
  print("The person is non-diabetic")
else:
  print("The person is diabetic")

# project finished