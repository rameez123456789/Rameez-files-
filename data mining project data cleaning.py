# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 16:12:45 2020

@author: Dell
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd




dataset= pd.read_excel(r"C:\Users\Dell\Desktop\train3_cleaned_Dorothy_only_LAT_left.xlsx")
d= pd.read_excel(r"C:\Users\Dell\Desktop\p1.xlsx")
d
dataset.head()
X = dataset.iloc[:, 1:9].values
X
y = dataset.iloc[:, 9].values
y
#gender
dummies = pd.get_dummies(dataset.Gender)
dummies

merged = pd.concat([dataset,dummies],axis=1)
merged

final = merged.drop(['Gender'], axis=1)
final
final = final.drop(['Gender'], axis=1)
final
#married
dummies = pd.get_dummies(final.Married)
dummies

merged1 = pd.concat([final,dummies],axis=1)
merged1

final1 = merged1.drop(['Married'], axis=1)
final1
final1 = final1.drop(['Married'], axis=1)
final1
#education
dummies = pd.get_dummies(final1.Education)
dummies

merged2 = pd.concat([final1,dummies],axis=1)
merged2

final2 = merged2.drop(['Education'], axis=1)
final2
final2 = final2.drop(['Education'], axis=1)
final2
#self employed
dummies = pd.get_dummies(final2.Self_Employed)
dummies

merged3 = pd.concat([final2,dummies],axis=1)
merged3

final3 = merged3.drop(['Self_Employed'], axis=1)
final3
final3 = final3.drop(['Self_Employed'], axis=1)
final3

final3 = final3.replace(['3+'],'3')

#dependent 


X1 = final3.drop(['Female','Dependents','Loan_ID','Loan_Amount_Term','Not Graduate','Property_Area','Loan_Status', 'No','No'],axis=1)
X1

y1 = final3.Loan_Amount_Term

final3= final3.dropna()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size = 0.25, random_state = 0)
print(X_train)
print(y_train)
print(X_test)
print(y_test)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train)
print(X_test)

# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

# Predicting a new result
#print(classifier.predict(sc.transform([[4695,0,96,1,1,0,1,1]])))
print(classifier.predict(sc.transform(d)))


a=(1828,1330,100,0,1,1,0)
for i in len(a):
     print(classifier.predict(sc.transform(a[i]))

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred

#print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)






from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size = 0.25, random_state = 0)

print(X_train)
print(y_train)
print(X_test)
print(y_test)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print(X_train)
print(X_test)

# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
y_pred
print(classifier.predict(sc.transform(d)))
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


from sklearn.linear_model import LinearRegression
model = LinearRegression()

model.fit(X1,y1)


model.predict(y1) # 2600 sqr ft home in new jersey

model.score(X1,y1)
#dummy=pd.get_dummies(dataset, columns=['Gender',''])

#dataset=pd.concat([dataset,dummy],axis=1)

#dataset=dataset.merge(dummy,left_index=True,right_index=True)
#dataset.head()