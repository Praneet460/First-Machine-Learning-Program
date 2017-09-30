from sklearn.datasets import load_iris
import numpy as np
from sklearn import tree

iris = load_iris()

#printing the features names in iris dataset
print (iris.feature_names)
#Results the features iris datasrt is having : ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

#printing the labels assigned in the iris dataset
print (iris.target_names)
#Results in labels assigned to the given dataset : ['setosa' 'versicolor' 'virginica']

#printing the specified iris dataset by their index
print (iris.data[0])
#Results in : [ 5.1  3.5  1.4  0.2]

print (iris.target[0])
#Results in : 0 i.e. 0 symbolises 'setosa'

#Remove one example of each label to use them in testing data set
#where  [0, 50, 100] are the indexes of different labels in iris dataset
test_idx = [0, 50, 100]   

#creating the training dataset to use in the classifier

train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

#creating the testing data for the classifier

test_target = iris.target[test_idx]
test_data = iris.data[test_idx] 

#train the classifier

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

#printing the testing data

print (test_target)

#results in the data we kept aside for testing: 
#[0 1 2]


#now comparing this data with the data predicted by the classifier we trained with the training data

print (clf.predict(test_data))

#Results in : [0 1 2] i.e. Our trained classifier is working fine


