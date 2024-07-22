import numpy
from scipy import stats #Only used for mode
import matplotlib.pyplot as plt

uniformDataSet = numpy.random.uniform(0.0, 5.0, 250) #Random dataset of floats from 0 to 5 inclusive

plt.hist(uniformDataSet, 100)
plt.show()

normalDataSet = numpy.random.normal(5.0, 1.0, 250) #Normal distriution dataset of floats where mean is 5, and std is 1

plt.hist(normalDataSet, 100)
plt.show()

#Scatterplots
x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

plt.scatter(x, y)
plt.show()

# Scaling data
# For each column, we can alter the data by using formula z = (x - u) / s
# x is the original value, u is the mean, and s is the standard deviation
# This turns the data into the amount of standard deviations from the mean
# Sklearn does this automatically

from sklearn.preprocessing import StandardScaler
import pandas

scale = StandardScaler()
df = pandas.read_csv("dataUnscaled.csv")
X = df[['Weight', 'Volume']]
scaledX = scale.fit_transform(X)
print(scaledX)
