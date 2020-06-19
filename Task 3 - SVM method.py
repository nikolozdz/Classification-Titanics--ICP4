import pandas as pd
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

dataset = pd.read_csv('glass.csv') # imports data
X = dataset.drop('Type', axis=1)
y = dataset['Type']

# Splitting the data for training 60% and testing 40%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state= 0)



classifier = SVC(kernel="linear") # Fitting Linear SVM to the Training set
classifier.fit(X_train, y_train)


y_pred = classifier.predict(X_test) # Predicting the Test set results

# Calculating SVC Accuracy
print("SVM Accuracy is: ",metrics.accuracy_score(y_test,y_pred)*100)
print(classification_report(y_test,y_pred))