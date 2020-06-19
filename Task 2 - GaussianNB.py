import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB

dataset = pd.read_csv('glass.csv') # Imports Data
X = dataset.drop('Type', axis=1)
y = dataset['Type'].values

# Splitting the data for training 60% and testing 40%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state= 0)


# Fitting Naive Bayes to the Training set
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

#Calculating Accuracy
print("Naive Bayes accuracy is: ", metrics.accuracy_score(y_test,y_pred)*100)
print(classification_report(y_test, y_pred))