# Some data is not contineous and is discrete, which a decision tree will help decide
# In this scenario, you are deciding whether or not to go to a comedy show


import pandas
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import sys

#Reads the data
df = pandas.read_csv("dataComedy.csv")

#Create a map from string to integer
d = {'UK': 0, 'USA': 1, 'N': 2}
df['Nationality'] = df['Nationality'].map(d)

d = {'YES': 1, 'NO': 0}
df['Go'] = df['Go'].map(d)

features=['Age', 'Experience', 'Rank', 'Nationality']
# Define feature columns and target columns
X = df[features]
y = df['Go']

dtree = DecisionTreeClassifier()
dtree = dtree.fit(X.values, y)

tree.plot_tree(dtree, feature_names=features)

# Predict the decision for a 40 year old American with 10 years of expirience, with a rank of 7
print(dtree.predict([[40, 10, 7, 1]]))
# Predict the decision for a 40 year old American with 10 years of expirience, with a rank of 6
print(dtree.predict([[40, 10, 6, 1]]))

# Keep in mind the decision tree is not set in stone, it can change beacuse its calculated using propabilities