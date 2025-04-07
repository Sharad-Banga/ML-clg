# Practical 6: Stimulate Naive Bayes Classifiers

# Data Preprocessing
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Creating a sample dataset instead of loading from CSV
data = {
    'User ID': np.arange(15624510, 15624510+400),
    'Gender': np.random.choice(['Male', 'Female'], size=400),
    'Age': np.random.randint(18, 70, size=400),
    'EstimatedSalary': np.random.randint(15000, 80000, size=400),
    'Purchased': np.random.randint(0, 2, size=400)
}

dataset = pd.DataFrame(data)

# Displaying the dataset
print(dataset)

# Extracting features and target variable
x = dataset.iloc[:, [2, 3]].values  # Age and EstimatedSalary
y = dataset.iloc[:, 4].values       # Purchased

# Splitting the dataset into training and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=0)

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

# Fitting the Naive Bayes classifier
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)

# Predicting the test set results
y_pred = classifier.predict(x_test)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)
print("\nAccuracy:", accuracy_score(y_test, y_pred))