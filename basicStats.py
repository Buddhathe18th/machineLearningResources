import numpy
from scipy import stats #Only used for mode

dataSet=[99,86,87,88,111,86,103,87,94,78,77,85,86]


print(numpy.mean(dataSet)) #Mean
print(numpy.median(dataSet)) #Median
print(stats.mode(dataSet)) #Mode

print(numpy.std(dataSet)) #Standard deviation: The range of the mean where most values lie
print(numpy.var(dataSet)) #Standard deviation: The range of the mean where most values lie
