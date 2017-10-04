#Predicting the type of car i.e. either 'sports-car' or 'minivan' based on two features i.e. car's horsepower and number of Seats

from sklearn import tree

#Collect training data

#features =[[horsepower, seats], [horsepower, seats], [horsepower, seats], [horsepower, seats]]
features =[[300, 2], [450, 2], [200, 8], [150, 9]]

#labels = [sports-car=0, sports-car=0, minivan=1, minivan=1]
labels = [0, 0, 1, 1]

#Train a Decision Tree Classifier

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

#Make Prediction

print (clf.predict([[300, 2]]))
#Result: [1] i.e. sports-car

print (clf.predict([[110, 10]]))
#Result: [0] i.e. minivan

print (clf.predict([[600, 1]]))
#Result: [0] i.e. sports-car

print (clf.predict([[450, 2]]))
#Result: [0] i.e. sports-car