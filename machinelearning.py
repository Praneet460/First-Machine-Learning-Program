from sklearn import tree

#Collect training data

#features =[[weight, smooth=1], [weight, smooth=1], [weight, bumpy=0], [weight, bumpy=0]]
features =[[140, 1], [130, 1], [150,0], [170,0]]

#labels = [apple=0, apple=0, orange=1, oragne=1]
labels = [0, 0, 1, 1]

#Train a Decision Tree Classifier

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features, labels)

#Make Prediction

print (clf.predict([[150, 0]]))

print (clf.predict([[110, 0]]))

print (clf.predict([[140, 1]]))

print (clf.predict([[100, 1]]))

