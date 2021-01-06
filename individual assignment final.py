
## Developing the model ###

# Load Librarie

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Import data
df = pd.read_excel(r"C:\Users\Dell\Desktop\Kickstarter.xlsx")
#heat map used to find gaps and empty spots
#sns.heatmap(df.isnull(), cbar=False)
df.info()

#data preperation
#droping last colown as it has less data
df = df.drop(['launch_to_state_change_days'],axis=1)
#droping all empty cells 
df = df.dropna()
#keeping only succcessful and failed
df = df.loc[(df['state'] == 'successful')|(df['state'] == 'failed')]
df1 = pd.get_dummies(df, columns = ['state','disable_communication',
                                    'country','currency','staff_pick','spotlight'])
#droping colowns 
df1.info()
cols = [41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60]
df1.drop(df1.columns[cols],axis=1,inplace=True)
df1.info()  

cols = [42,43,44,45,46,47,48,49,50,51,52,53,55,57]
df1.drop(df1.columns[cols],axis=1,inplace=True)

df1.info()


#creating Tech catergories 
df1 = df1.assign(Tech = (df.category == 'Apps')|(df.category == 'Flights')|
               (df.category == 'Gadgets')|(df.category == 'Hardware')|
               (df.category == 'Makerspaces')|(df.category == 'Sound')|
               (df.category == 'Robots')|(df.category == 'Wearables')|
               (df.category == 'Web')|(df.category == 'Software'))

df1.info()
#creating weekend and weeekday catergories 
df1 = df1.assign(deadline_weekend1 = (df.deadline_weekday == 'Sunday')|(df.deadline_weekday == 'Saturday'))
df1 = df1.assign(state_changed_at_weekday1 = (df.state_changed_at_weekday == 'Sunday')|(df.state_changed_at_weekday== 'Saturday'))
df1 = df1.assign(created_at_weekday1 = (df.created_at_weekday == 'Sunday')|(df.created_at_weekday== 'Saturday'))
df1 = df1.assign(launched_at_weekday1 = (df.launched_at_weekday == 'Sunday')|(df.launched_at_weekday == 'Saturday'))

#droping 
 
df1.info()
df1= df1.drop(['category','deadline_weekday','state_changed_at_weekday','created_at_weekday','launched_at_weekday'],axis=1)
df1= df1.drop(['project_id','name','created_at','launched_at','deadline','state_changed_at'],axis=1)
              
df1.info()
#dummy catergorical variable
df1 = pd.get_dummies(df1, columns = ['Tech','deadline_weekend1',
                                    'state_changed_at_weekday1','created_at_weekday1','launched_at_weekday1'])

df1.info()

cols = [32,34,36,38,40,42]
df1.drop(df1.columns[cols],axis=1,inplace=True)
df1.info()  


#dealing with anomalies 

# Create isolation forest model
from sklearn.ensemble import IsolationForest
iforest=IsolationForest(n_estimators=100,contamination=.04,random_state=0)

pred=iforest.fit_predict(df1)
score=iforest.decision_function(df1)
pred

# Extracting anomalies
from numpy import where
anom_index=where(pred==-1)
values=df1.iloc[anom_index]
values

#list of anomalies values 
v=list(anom_index)
v

from numpy import where
anom_index=where(pred==-1)
values=df1.iloc[anom_index]
values

df1=pd.concat([df1,values]).drop_duplicates(keep=False)
df1.info

###########################################################################################
#Regression 

X = df1.drop(['usd_pledged'],axis=1)
X1=df1.drop(['usd_pledged','deadline_yr','state_changed_at_month',
             'state_changed_at_day','state_changed_at_yr','spotlight_True','disable_communication_False'
             ,'name_len','blurb_len',
             'state_changed_at_hr','created_at_yr','state_failed','state_successful',
             'backers_count','pledged',
             'state_changed_at_weekday1_True'],axis=1)
y = df1["usd_pledged"]
X1.info()


#finding corealtion and droping required features that are colinear 
corr_matrix = X1.corr(method= 'pearson')
print(corr_matrix)
#fetures selected
#   Column                     Non-Null Count  Dtype  
#---  ------                     --------------  -----  
# 0   goal                       4857 non-null   float64
# 1   static_usd_rate            4857 non-null   float64
# 2   name_len_clean             4857 non-null   float64
# 3   blurb_len_clean            4857 non-null   float64
# 4   deadline_month             4857 non-null   int64  
# 5   deadline_day               4857 non-null   int64  
# 6   deadline_hr                4857 non-null   int64  
# 7   created_at_month           4857 non-null   int64  
# 8   created_at_day             4857 non-null   int64  
# 9   created_at_hr              4857 non-null   int64  
# 10  launched_at_month          4857 non-null   int64  
# 11  launched_at_day            4857 non-null   int64  
# 12  launched_at_yr             4857 non-null   int64  
# 13  launched_at_hr             4857 non-null   int64  
# 14  create_to_launch_days      4857 non-null   int64  
# 15  launch_to_deadline_days    4857 non-null   int64  
# 16  country_US                 4857 non-null   uint8  
# 17  currency_USD               4857 non-null   uint8  
# 18  Tech_True                  4857 non-null   uint8  
# 19  deadline_weekend1_True     4857 non-null   uint8  
# 20  created_at_weekday1_True   4857 non-null   uint8  
# 21  launched_at_weekday1_True  4857 non-null   uint8 

#standarization process
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_std= scaler.fit_transform(X1)

##############################################################################
#Feature selection in lasso and alpha value .05

from sklearn.linear_model import Lasso
model = Lasso(alpha=.05)
model.fit(X_std,y)
model.coef_
pd.DataFrame((zip(X1.columns,model.coef_)), columns = ['predictor','coefficient'])

#post removal coefficients 
#predictor   coefficient
#0                        goal  3.475721e+02
#1             static_usd_rate  1.027724e+03
#2              name_len_clean  6.383096e+03
#3             blurb_len_clean  7.430468e+02
#4              deadline_month  6.870774e+02
#5                deadline_day -9.556384e+02
#6                 deadline_hr  2.309806e+03
#7            created_at_month -9.726802e+02
#8              created_at_day -1.313311e+03
#9               created_at_hr -2.514813e+02
#10          launched_at_month  2.703775e+01
#11            launched_at_day -1.271426e+03
#12             launched_at_yr  2.605805e+02
#13             launched_at_hr -6.733565e+03
#14      create_to_launch_days  1.072681e+03
#15    launch_to_deadline_days  1.591446e+03
#16                 country_US  5.074907e+03
#17               currency_USD  3.067930e-13
#18                  Tech_True  5.332839e+03
#19     deadline_weekend1_True -3.808377e+03
#20   created_at_weekday1_True -2.680603e+03
#21  launched_at_weekday1_True -1.347240e+03

#Lasso MSE:2583930640.734356
# Split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size = 0.30, random_state = 7)

# Load libraries
from sklearn.linear_model import Lasso

# Separate the data
from sklearn.model_selection import train_test_split

# Run Lasso
lasso1=Lasso()
model2=lasso1.fit(X_train,y_train)

# Generate the prediction value from the test data
y_test_pred=model2.predict(X_test)

# Calculate the MSE
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_test,y_test_pred)
print(mse)
#Alpha =  8  / MSE = 1462523256.2550206
# Run the loops
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
for i in range (1,10):
     lasso3 = Lasso(alpha=i)
     model3 = lasso3.fit(X_train,y_train)
     y_test_pred = model3.predict(X_test)
     print('Alpha = ',i,' / MSE =',mean_squared_error(y_test, y_test_pred))
     #model3.coef_
     #print(pd.DataFrame((zip(X.columns,model3.coef_)), columns = ['predictor','coefficient']))


#RandomForestRegressor MSE:1519352053.7245193
# Load libraries
from sklearn.ensemble import RandomForestRegressor
import pandas

# Separate the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size = 0.30, random_state = 5)
# Run Random Forest
rf = RandomForestRegressor(random_state=0, n_estimators=100)
model4 = rf.fit(X_train,y_train)
# Using the model to predict the results based on the test dataset
y_test_pred = model4.predict(X_test)
# Calculate the mean squared error of the prediction
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_test_pred)
print(mse)



#GradientBoosting REgressor MSE:1384225142.539976
# Load libraries
from sklearn.ensemble import GradientBoostingRegressor
import pandas
# Separate the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size = 0.33, random_state = 5)
# Run Random Forest
gbt = GradientBoostingRegressor(random_state=0, n_estimators=150)
model5 = gbt.fit(X_train,y_train)
# Using the model to predict the results based on the test dataset
y_test_pred = model5.predict(X_test)
# Calculate the mean squared error of the prediction
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_test_pred)
print(mse)


#Randomforestregressor K fold MSE: max features 4 : -1691612409.3591785
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
for i in range (2,7):
     model2 = RandomForestRegressor(random_state=5,max_features=i,n_estimators=700)
     scores = cross_val_score(estimator=model2, X=X1, y=y, cv=5,scoring = 'neg_mean_squared_error')
     print(i,':',np.average(scores))     




 
#############################################################################################

#classification
X2=df1.drop(['state_successful','state_failed','deadline_yr','state_changed_at_month',
             'state_changed_at_day','state_changed_at_yr','spotlight_True',
             'name_len','blurb_len','disable_communication_False',
             'state_changed_at_hr','created_at_yr','state_failed','backers_count','pledged',
             'usd_pledged',
             'state_changed_at_weekday1_True'],axis=1)
y2 = df1["state_successful"]
X2.info()
##features selected 
# 0   goal                       4857 non-null   float64
# 1   static_usd_rate            4857 non-null   float64
# 2   name_len_clean             4857 non-null   float64
# 3   blurb_len_clean            4857 non-null   float64
# 4   deadline_month             4857 non-null   int64  
# 5   deadline_day               4857 non-null   int64  
# 6   deadline_hr                4857 non-null   int64  
# 7   created_at_month           4857 non-null   int64  
# 8   created_at_day             4857 non-null   int64  
# 9   created_at_hr              4857 non-null   int64  
# 10  launched_at_month          4857 non-null   int64  
# 11  launched_at_day            4857 non-null   int64  
# 12  launched_at_yr             4857 non-null   int64  
# 13  launched_at_hr             4857 non-null   int64  
# 14  create_to_launch_days      4857 non-null   int64  
# 15  launch_to_deadline_days    4857 non-null   int64  
# 16  country_US                 4857 non-null   uint8  
# 17  currency_USD               4857 non-null   uint8  
# 18  Tech_True                  4857 non-null   uint8  
# 19  deadline_weekend1_True     4857 non-null   uint8  
# 20  created_at_weekday1_True   4857 non-null   uint8  
# 21  launched_at_weekday1_True  4857 non-null   uint8 

#standarization process
from sklearn.preprocessing import StandardScaler
import numpy
scaler = StandardScaler()
X_std1= scaler.fit_transform(X2)     
 
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_std1, y2, test_size = 0.30, random_state = 5)    

############################################################################################
#RandomForestClassifier:K fold method at 4 cross val score:4 : 0.7488143825253335
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
for i in range (2,7):
     model2 = RandomForestClassifier(random_state=5,max_features=i,n_estimators=500)
     scores = cross_val_score(estimator=model2, X=X2, y=y2, cv=5)
     print(i,':',numpy.average(scores))     

############################################################################################
#logistic Regresssion method: Accuracy Score:0.7061759201497193,MSE:0.29382407985028075    
# Load libraries
from sklearn.linear_model import LogisticRegression

# Split the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_std1, y2, test_size = 0.33, random_state = 5)

# Run the model
lr2 = LogisticRegression(max_iter=1000)
model3 = lr2.fit(X_train,y_train)
# Calculate the accuracy score
from sklearn import metrics
y_test_pred = model3.predict(X_test)
print(metrics.accuracy_score(y_test, y_test_pred))
# Print the confusion matrix
metrics.confusion_matrix(y_test, y_test_pred)
# Confusion matrix with label
print(pandas.DataFrame(metrics.confusion_matrix(y_test, y_test_pred, labels=[0,1]), index=['true:0',
'true:1'], columns=['pred:0', 'pred:1']))
metrics.precision_score(y_test, y_test_pred)
metrics.recall_score(y_test, y_test_pred)
# Calculate the F1 score
metrics.f1_score(y_test, y_test_pred)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_test_pred)
print(mse)

##############################################################################################

#GradientBoostingClassifier:Accuracy Score:0.7442295695570805,MSE:0.2557704304429195
# Load libraries
from sklearn.ensemble import GradientBoostingClassifier
import pandas
# Separate the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_std1, y2, test_size = 0.33, random_state = 5)
# Run Random Forest
gbt = GradientBoostingClassifier(random_state=0, n_estimators=150)
model6 = gbt.fit(X_train,y_train)
# Using the model to predict the results based on the test dataset
y_test_pred = model6.predict(X_test)
# Calculate the mean squared error of the prediction
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_test_pred)
print(mse)
print(metrics.accuracy_score(y_test, y_test_pred))

###############################################################################################

#RandomForestClassifier:Accuracy Score:0.7361197754210854,MSE:0.2638802245789145
# Load libraries
from sklearn.ensemble import RandomForestClassifier
import pandas

# Separate the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size = 0.33, random_state = 5)
# Run Random Forest
rf = RandomForestClassifier(random_state=0, n_estimators=100)
model4 = rf.fit(X_train,y_train)
# Using the model to predict the results based on the test dataset
y_test_pred = model4.predict(X_test)
# Calculate the mean squared error of the prediction
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_test_pred)
print(mse)
print(metrics.accuracy_score(y_test, y_test_pred))


##################################################################################################

#Problem statement is wanted to cluster and know how the kichstarter campaigns work 
# using goal ,backers count,usd_pledged,create_to_launch_days ,launch_to_deadline_days,
#  state_successful ,currency_USD ,country_US 


X11=df1.drop(['deadline_yr','state_changed_at_month','state_failed',
              'country_US',
             'state_changed_at_day','state_changed_at_yr','spotlight_True','static_usd_rate',
             'name_len','blurb_len','name_len_clean','disable_communication_False',
             'state_changed_at_hr','created_at_yr','blurb_len_clean'
             ,'pledged','deadline_month','deadline_day','deadline_hr','created_at_month',
             'created_at_day','created_at_hr','launched_at_month','launched_at_day',
             'launched_at_yr','launched_at_hr'
             ,'deadline_weekend1_True','created_at_weekday1_True',
             'launched_at_weekday1_True',
             'state_changed_at_weekday1_True'],axis=1)
#

X11.info()
from sklearn.preprocessing import StandardScaler
import numpy
scaler = StandardScaler()
X_std11= scaler.fit_transform(X11) 

from sklearn.cluster import KMeans
withinss = []
for i in range (2,8):
     kmeans = KMeans(n_clusters=i)
     model = kmeans.fit(X)
     withinss.append(model.inertia_)
from matplotlib import pyplot
pyplot.plot([2,3,4,5,6,7],withinss)

#as we can see from graph 6 will be ideal number of clusters 
#according to interia calculation 

#using Kmeans clustering method 
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=6)
model = kmeans.fit(X_std11)
labels = model.labels_
from sklearn.metrics import silhouette_samples
silhouette = silhouette_samples(X_std11,labels)
import pandas
df = pandas.DataFrame({'label':labels,'silhouette':silhouette})
print('Average Silhouette Score for Cluster 0: ',np.average(df[df['label'] == 0].silhouette))
print('Average Silhouette Score for Cluster 1: ',np.average(df[df['label'] == 1].silhouette))
print('Average Silhouette Score for Cluster 2: ',np.average(df[df['label'] == 2].silhouette))
print('Average Silhouette Score for Cluster 3: ',np.average(df[df['label'] == 3].silhouette))
print('Average Silhouette Score for Cluster 4: ',np.average(df[df['label'] == 4].silhouette))
print('Average Silhouette Score for Cluster 5: ',np.average(df[df['label'] == 5].silhouette))

from sklearn.metrics import silhouette_score
silhouette_score(X_std11,labels)

a=df[df['label'] == 0]
a.describe(include='all').transpose()

b=df[df['label'] == 1]
b.describe(include='all').transpose()
c=df[df['label'] == 2]
c.describe(include='all').transpose()
d=df[df['label'] == 3]
d.describe(include='all').transpose()
e=df[df['label'] == 4]
e.describe(include='all').transpose()
f=df[df['label'] == 5]
f.describe(include='all').transpose()

#centro of the clusters 
centers = model.cluster_centers_
centers 
#labeling the cluster with coefficients of predictors of various cluters .  
centers = pd.DataFrame(model.cluster_centers_, columns = X11.columns)   
centers  


## Grading section validation of model ##


#kickstarter_grading_df = pandas.read_excel("Kickstarter-Grading.xlsx")
#kickstarter_grading_df = kickstarter_grading_df.dropna()

#X_grading = kickstarter_grading_df[["name_len","blurb_len"]]
#y_grading = kickstarter_grading_df["state"]

#y_grading_pred = model.predict(X_grading)

#accuracy_score(y_grading, y_grading_pred)


#Grading data preperation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Import data
df11 = pd.read_excel(r"C:\Users\Dell\Desktop\Kickstarter-Grading-Sample (1).xlsx")
#sns.heatmap(df11.isnull(), cbar=False)
df11.info()

df11 = df11.drop(['launch_to_state_change_days'],axis=1)
df11 = df11.dropna()
df11 = df11.loc[(df11['state'] == 'successful')|(df11['state'] == 'failed')]
df12 = pd.get_dummies(df11, columns = ['state','disable_communication',
                                    'country','currency','staff_pick','spotlight'])

df1.info()
cols = [41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60]
df12.drop(df12.columns[cols],axis=1,inplace=True)
df12.info()  

cols = [42,43,44,45,46,47,48,49,50,51,52,53,55,57]
df12.drop(df12.columns[cols],axis=1,inplace=True)

df12.info()

df12 = df12.assign(Tech = (df12.category == 'Apps')|(df12.category == 'Flights')|
               (df12.category == 'Gadgets')|(df12.category == 'Hardware')|
               (df12.category == 'Makerspaces')|(df12.category == 'Sound')|
               (df12.category == 'Robots')|(df12.category == 'Wearables')|
               (df12.category == 'Web')|(df12.category == 'Software'))

df12.info()

df12 = df12.assign(deadline_weekend1 = (df12.deadline_weekday == 'Sunday')|(df12.deadline_weekday == 'Saturday'))
df12 = df12.assign(state_changed_at_weekday1 = (df12.state_changed_at_weekday == 'Sunday')|(df12.state_changed_at_weekday== 'Saturday'))
df12 = df12.assign(created_at_weekday1 = (df12.created_at_weekday == 'Sunday')|(df12.created_at_weekday== 'Saturday'))
df12 = df12.assign(launched_at_weekday1 = (df12.launched_at_weekday == 'Sunday')|(df12.launched_at_weekday == 'Saturday'))
 
df12.info()
df12= df12.drop(['category','deadline_weekday','state_changed_at_weekday','created_at_weekday','launched_at_weekday'],axis=1)
df12= df12.drop(['project_id','name','created_at','launched_at','deadline','state_changed_at'],axis=1)
              
df12.info()

df12 = pd.get_dummies(df12, columns = ['Tech','deadline_weekend1',
                                    'state_changed_at_weekday1','created_at_weekday1','launched_at_weekday1'])

df12.info()

cols = [32,34,36,38,40,42]
df12.drop(df12.columns[cols],axis=1,inplace=True)
df12.info()  



# Create isolation forest model
from sklearn.ensemble import IsolationForest
iforest=IsolationForest(n_estimators=100,contamination=.04,random_state=0)

pred=iforest.fit_predict(df12)
score=iforest.decision_function(df12)
pred

# Extracting anomalies
from numpy import where
anom_index=where(pred==-1)
values=df12.iloc[anom_index]
values

#list of anomalies values 
v=list(anom_index)
v

from numpy import where
anom_index=where(pred==-1)
values=df12.iloc[anom_index]
values

df12=pd.concat([df12,values]).drop_duplicates(keep=False)
df12.info


#Regression testing
X12=df12.drop(['usd_pledged','deadline_yr','state_changed_at_month',
             'state_changed_at_day','state_changed_at_yr','spotlight_True','disable_communication_False'
             ,'name_len','blurb_len',
             'state_changed_at_hr','created_at_yr','state_failed','state_successful',
             'backers_count','pledged',
             'state_changed_at_weekday1_True'],axis=1)
y12= df12["usd_pledged"]
X12.info()
#standardization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_std12= scaler.fit_transform(X12)

from sklearn.ensemble import GradientBoostingRegressor

# Run GBT testing 
#gbt = GradientBoostingRegressor(random_state=0, n_estimators=150)
#model5 = gbt.fit(X_train,y_train)
# Using the model to predict the results based on the test dataset
y_test_pred = model5.predict(X_std12)
# Calculate the mean squared error of the prediction
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y12, y_test_pred)
print(mse)


#classification testing
X22=df12.drop(['state_successful','state_failed','deadline_yr','state_changed_at_month',
             'state_changed_at_day','state_changed_at_yr','spotlight_True',
             'name_len','blurb_len','disable_communication_False',
             'state_changed_at_hr','created_at_yr','state_failed','backers_count','pledged',
             'usd_pledged',
             'state_changed_at_weekday1_True'],axis=1)
y22 = df12["state_successful"]
X22.info()


#GradientBoostingClassifier:
# Load libraries
from sklearn.ensemble import GradientBoostingClassifier
import pandas
# Separate the data
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X_std1, y2, test_size = 0.33, random_state = 5)
# Run Random Forest
#gbt = GradientBoostingClassifier(random_state=0, n_estimators=150)
#model6 = gbt.fit(X_train,y_train)
# Using the model to predict the results based on the test dataset
y_test_pred = model6.predict(X22)
# Calculate the mean squared error of the prediction
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y22, y_test_pred)
print(mse)
print(metrics.accuracy_score(y22, y_test_pred))

