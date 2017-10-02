#Work on iris dataset
#importing the iris dataset in the program

from sklearn import datasets
iris = datasets.load_iris()

#assigning features = X
X = iris.data

#assigning labels = Y
Y = iris.target

#distributing the whole dataset in equal halfs into training and testing datasets

from sklearn.cross_validation import train_test_split

#X_train and Y_train are the features and labels for the training data
#X_test and Y_test are the features and labels for the testing data
#test_size = 0.5 i.e. split the entire dataset into equal halfs
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.5)

#now we create the classifier
from sklearn import tree
my_clf = tree.DecisionTreeClassifier()

#used the training data to train the classifier
my_clf.fit(X_train, Y_train)

#test the trained classifier on the testing dataset
prediction = my_clf.predict(X_test)
print (prediction)

#printing the testing dataset we splited before from the entire dataset 
print (Y_test)

 
#now checking the accuracy of our trained classifier
from sklearn.metrics import accuracy_score
print (accuracy_score(Y_test, prediction))