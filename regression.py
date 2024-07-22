import numpy
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score
import pandas
from sklearn import linear_model

#Linear regression
x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]

slope, intercept, r, p, std_err = stats.linregress(x, y)

def line(x): #Regression function, using this we can predict future values
  return slope * x + intercept #Takes any x input and outputs the y coordinate on the line

mymodel = list(map(line, x)) #Maps the input x to the regressed y coordinate

plt.scatter(x, y)

plt.plot(x, mymodel) #Connects the points with a straight line, in this case, it will be one long straight line
plt.show()


#Polynomial regression
x = [1,2,3,5,6,7,8,9,10,12,13,14,15,16,18,19,21,22]
y = [100,90,80,60,60,55,60,65,70,70,75,76,78,79,90,99,99,100]

mymodel = numpy.poly1d(numpy.polyfit(x, y, 3)) #Creates a 3rd degree polynomial to fit the data

myline = numpy.linspace(1, 22, 100) #Array of values from 1 to 22, with 100 points in total

plt.scatter(x, y)
plt.plot(myline, mymodel(myline)) #Draws the polynomial
plt.show()
print(r2_score(y, mymodel(x))) #r^2 score for this fit, showing how close the polynomial fits the data


#Multiple regression

df = pandas.read_csv("dataScaled.csv") #Data file

X = df[['Weight', 'Volume']] #Independant variables
y = df['CO2'] #Dependant variables

regr = linear_model.LinearRegression() #Creates a linear regression object
regr.fit(X, y) #Fits the linear regression object onto the data set

print(regr.predict([[2300, 1300]])) #Predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300cm3:
print(regr.coef_) #Coefficients on the independant variables