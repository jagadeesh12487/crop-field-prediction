#!/usr/bin/env python
# coding: utf-8

# In[8]:


#import libraries for access and functional purpose
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df_input=pd.read_csv(r"Rice crop.csv")
df_input.shape
df_input.head()
df_input['Crop'].nunique()
df_input['Crop'].value_counts()
df_input['District_name'].nunique()
df_input['District_name'].value_counts()
df_input['Season'].nunique()
df_input['Season'].value_counts()
df_input['Crop_Year'].nunique()
df_input['Crop_Year'].value_counts()
df_input.info()
df_input.isnull().sum()
df_input.duplicated().sum()
df_input = df_input[df_input['Area'] > 0 ]
df_input = df_input[df_input['Production'] > 0 ]
df_input = df_input.replace(r'^\s*$', np.NaN, regex=True)
df_input = df_input.dropna()
df_input["ProductionPerArea"] = ((df_input["Production"])/(df_input["Area"]))
df_input = df_input.drop(columns=['State_Name','Area','Production'])
Crops = df_input['Crop']
CropsCount = {}
for crop in Crops:
    CropsCount[crop] = CropsCount.get(crop, 0)+1
#extract values and keys of dict:CropsCount
27
labels = list(CropsCount.keys())
values = list(CropsCount.values())
plt.pie(values, labels=labels,autopct='%1.1f%%')
plt.show()
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
categorical_columns = ['District_name', 'Crop' ,'Season']
#label encoder dict
labels_dict = {}
#scaling dict
scaling_dict = {}
for column in categorical_columns:
    le = LabelEncoder()
    le.fit(df_input[column])
    df_input[column] = le.transform(df_input[column])
    labels_dict[column] = le.classes_
    df_input.describe()
    x = df_input['ProductionPerArea']
    print(x.describe())
import matplotlib.pyplot as plt
plt.style.use('ggplot')
fig, ax = plt.subplots()
ax.boxplot((x), vert=False, showmeans=True, meanline=True,
labels=('x'), patch_artist=True,
medianprops={'linewidth': 2, 'color': 'purple'},
meanprops={'linewidth': 2, 'color': 'red'})
plt.show()
def deviationTransform(arr):
    d = np.std(arr)
    return [0,d]
def minMaxTransform(arr):
    min = np.min(arr)
    28
    max = np.max(arr)
    return [min,max-min]
for column in categorical_columns:
    scaling_params = minMaxTransform(np.array(df_input[column]))
df_input[column] = (df_input[column] -
scaling_params[0])/scaling_params[1]
scaling_dict[column] = scaling_params
scaling_params = minMaxTransform(np.array(df_input['Crop_Year']))
scaling_dict['Crop_Year'] = scaling_params
df_input['Crop_Year'] = (df_input['Crop_Year'] -
scaling_params[0])/scaling_params[1]
scaling_params = deviationTransform(np.array(df_input['ProductionPerArea']))
df_input['ProductionPerArea'] = (df_input['ProductionPerArea'] -
scaling_params[0])/scaling_params[1]
scaling_dict['ProductionPerArea'] = scaling_params
p1 = np.percentile(np.array(df_input['ProductionPerArea']), 25)
p2 = np.percentile(np.array(df_input['ProductionPerArea']), 99)
df_input = df_input[df_input['ProductionPerArea'] > p1]
df_input = df_input[df_input['ProductionPerArea'] < p2]
from sklearn.model_selection import train_test_split
df_small = df_input
df_small.columns.name = None
df=df_small
x_train, x_test, y_train, y_test = train_test_split(df.iloc[:,:-1], df.iloc[:,-1],
test_size=0.2, random_state=2)
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import r2_score
def trained(clf, x_train, x_test, y_train, y_test):
    clf.fit(x_train, y_train)
    y_train_pred = clf.predict(x_train)
    y_pred = clf.predict(x_test)  # Calculate y_pred here
    print("MSE " + str(mean_squared_error(y_train, y_train_pred)))
    print("MAE " + str(mean_absolute_error(y_train, y_train_pred)))
    print("RMSE " + str(np.sqrt(mean_squared_error(y_train, y_train_pred))))
    print("RMSLE " + str(np.sqrt(mean_squared_log_error(y_test, y_pred))))  # Move this line here
    R2_score = r2_score(y_train, y_train_pred)
    print("R2 Score " + str(R2_score))
    y_pred = np.array(y_train_pred)
    y_test = np.array(y_train)
    data = {}
    data['x'] = y_test
    data['y'] = y_pred
    sns.regplot(x="x", y="y", data=data)

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import r2_score
def tested(clf,x_train,x_test,y_train,y_test):
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    print("MSE " + str(mean_squared_error(y_test, y_pred)))
    print("MAE " + str(mean_absolute_error(y_test, y_pred)))
    print("RMSE " + str(np.sqrt(mean_squared_error(y_test, y_pred))))
    print("RMSLE " + str(np.sqrt(mean_squared_log_error(y_test, y_pred))))
    R2_score = r2_score(y_test,y_pred)
    print("R2 Score " + str(R2_score))
    y_pred = np.array(y_pred)
    y_test = np.array(y_test)
    data={}
    data['x'] = y_test
    data['y'] = y_pred
    sns.regplot(x="x",y="y",data=data);
    
#RandomForest    
from sklearn.ensemble import RandomForestRegressor
regresser = RandomForestRegressor(n_estimators = 10 ,random_state = 0)
print("\t\t\t random-forest classifier")
trained(regresser,x_train,x_test,y_train,y_test)
from sklearn.ensemble import RandomForestRegressor
regresser = RandomForestRegressor(n_estimators = 10 ,random_state = 0)
print("\t\t\t random-forest classifier")
tested(regresser,x_train,x_test,y_train,y_test)
from sklearn.ensemble import RandomForestRegressor
regresser = RandomForestRegressor(n_estimators = 10 ,random_state = 0)
regresser.fit(x_train,y_train)
y_pred = regresser.predict(x_test)
y_pred = np.array(y_pred)
y_test = np.array(y_test)
for i in range(y_test.size):
    if(y_test[i]>0 and y_test[i]<0.001):
        print(i,y_test[i],y_pred[i])
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import r2_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
def classify(clf,x_train,x_test,y_train,y_test):
    clf.fit(x_train,y_train)
    y_pred = clf.predict(x_test)
    print(y_pred)
    y_train_pred = clf.predict(x_train)
    metricsList = []
    metricsList.append(mean_squared_error(y_train, y_train_pred))
    metricsList.append(mean_absolute_error(y_test, y_pred))
    metricsList.append(np.sqrt(mean_squared_error(y_test, y_pred)))
    metricsList.append(r2_score(y_test,y_pred))
    return metricsList
clfMetrics = []
from sklearn.model_selection import train_test_split
df_small = df_input
df_small.columns.name = None
df=df_small
x_train, x_test, y_train, y_test = train_test_split(df.iloc[:,:-1], df.iloc[:,-1],
test_size=0.2, random_state=2)
from sklearn.ensemble import RandomForestRegressor
regresser = RandomForestRegressor(n_estimators = 10 ,random_state = 0)
print("\t\t\t random-forest classifier")
clfMetrics.append((classify(regresser,x_train,x_test,y_train,y_test)))
print(clfMetrics)
print(clfMetrics)


# In[ ]:




