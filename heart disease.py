import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Loading the dataset to a pandas DataFrame
heart_data = pd.read_csv("C:/Users/ankus/Desktop/Heart Disease/heart.csv")

# print first 5 rows of the dataset
print(heart_data.head())

#print last 5 rows of the dataset
print(heart_data.tail())

# number of rows and columns in the dataset
print(heart_data.shape)

# getting some info about the dataset
print(heart_data.info())

# checking for missing values
print(heart_data.isnull().sum())

# statistical measures about the dataset
print(heart_data.describe())

# checking the distribution of Target Variable
print(heart_data['target'].value_counts())

# 1 --> Defective Heart
# 0 --> Healthy Heart

# Plot target variable distribution
heart_data['target'].value_counts().plot(kind='bar', color=['green','red'])
plt.title("Distribution of Heart Disease (1 = Yes, 0 = No)")
plt.xlabel("Heart Disease")
plt.ylabel("Count")
plt.show()



# Splitting the features and target
X = heart_data.drop(columns = 'target', axis=1)
Y = heart_data['target']
print(X)
print(Y)


# Splitting the data into training data and test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

print(X.shape, X_train.shape, X_test.shape)

# Model Training
model = LogisticRegression()

# training the Logistic Regression model with Training data
model.fit(X_train , Y_train)

# Model Evaluation
# accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)

print('Accuracy on Training data : ', training_data_accuracy)

# accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)

print('Accuracy on Test data : ', test_data_accuracy)

# Building a Predictive System
input_data = (63,0,2,135,252,0,0,172,0,0,2,0,2)

# changing the input_data to numpy array
input_numpy = np.asarray(input_data)

# reshaping the array as we are predicting for one instance
input_numpy_reshaped = input_numpy.reshape(1, -1)

prediction = model.predict(input_numpy_reshaped)
print(prediction)

if (prediction[0] == 0):
    print('The person does not have a Heart Disease')
else:
    print('The person has Heart Disease')


