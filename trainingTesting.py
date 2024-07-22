# In training and testing, we will split our data set into two sets, 80% goes to training and 20% goes to testing

# We will mimic 100 customers and their shopping habits

import numpy
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
numpy.random.seed(2)

x = numpy.random.normal(3, 1, 100) # Number of minutes before a purchase
y = numpy.random.normal(150, 40, 100) / x # Money spent on a purchase

#Splitting given data
train_x = x[:80]
train_y = y[:80]

test_x = x[80:]
test_y = y[80:]

#Training set
plt.scatter(train_x, train_y)
plt.show()

#Testing set
plt.scatter(test_x, test_y)
plt.show()

#Create a fit with polynomial regression
mymodel = numpy.poly1d(numpy.polyfit(train_x, train_y, 4))

myline = numpy.linspace(0, 6, 100)

plt.scatter(train_x, train_y)
plt.plot(myline, mymodel(myline))
plt.show()

# Looking at the end of the line of best fit, the line tilts up at the last minute, which might be a sign of overfitting

print(r2_score(train_y,mymodel(train_x)))

# An r^2 score of 0.799 means its not a bad fit, but isn't perfect

print(r2_score(test_y,mymodel(test_x)))

# An r^2 score of 0.809 on the testing data means the model is not bad, and can be used to predict future data