import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt

#importing the dataset
df = pd.read_csv(r"Social_Network_Ads.csv")
X = df.iloc[:, :-1].values #creating an array of the first two columns
Y = df.iloc[:,-1].values #creating an array of the last column

#Splitting the dataset into training set and test set: 30%,70% test train split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.32, random_state = 0) 

#feature scaling to normalize the range
fs = StandardScaler()
X_train = fs.fit_transform(X_train) 
X_test = fs.transform(X_test)

#creating a decision tree classifier
clf = DecisionTreeClassifier(criterion='entropy', random_state=0)
clf.fit(X_train, Y_train)

#prediction
Y_pred = clf.predict(X_test)

#confusion matrix
cm = confusion_matrix(Y_test, Y_pred)
print(cm)
accuracy_score(Y_test, Y_pred)

#displaying the test values
plt.figure(figsize = (12,8))
tree.plot_tree(clf.fit(X_test, Y_test))
plt.show()







