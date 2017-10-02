#Testing the features for making better classifier prediction

import numpy as np
import matplotlib.pyplot as plt

#taking total 1000 examples
#total 500 examples of heights of greyhounds breed of dogs
greyhounds = 500

#total 500 examples of heights of labrador breed of dogs
labs = 500

#let greyhounds are average 28 inches tall
#and their height varies with 4 inches
grey_height = 28 + 4 * np.random.randn(greyhounds)
print (grey_height)
#let labradors are average 24 inches tall
#and their height varies with 4 inches
lab_height = 24 + 4 * np.random.randn(labs)
print (lab_height)

#Visualize these two array of greyhounds and labradors heights in a histogram
#represent greyhounds with red and labradors with blue color
plt.hist([grey_height, lab_height], stacked=True, color=['r','b'])
plt.show()
