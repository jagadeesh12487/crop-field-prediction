#!/usr/bin/env python
# coding: utf-8

# In[2]:


import matplotlib.pyplot as plt
plt.plot([5,10,15,20])
plt.plot([24,23,21,5])
plt.plot([1,2,3,4,5])
plt.show()


# In[2]:


# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load your dataset (replace 'data.csv' with your dataset)
data = pd.read_csv('data.csv')

# Data preprocessing and feature engineering (adjust as needed)
# Example: Selecting relevant features and target variable
X = data[['Weather', 'Soil_Type', 'Fertilizer', 'Planting_Density']]
y = data['Crop_Yield']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train a Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)  # You can adjust hyperparameters
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print('Model Evaluation:')
print(f'Mean Absolute Error: {mae}')
print(f'Mean Squared Error: {mse}')
print(f'Root Mean Squared Error: {rmse}')
print(f'R-squared (R2) Score: {r2}')

# Hyperparameter tuning (you can adjust the parameter grid)
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_rf_model = grid_search.best_estimator_

# Make predictions using the best model
y_pred_best = best_rf_model.predict(X_test)

# Evaluate the best model
mae_best = mean_absolute_error(y_test, y_pred_best)
mse_best = mean_squared_error(y_test, y_pred_best)
rmse_best = np.sqrt(mse_best)
r2_best = r2_score(y_test, y_pred_best)

print('Best Model Evaluation:')
print(f'Mean Absolute Error (Best Model): {mae_best}')
print(f'Mean Squared Error (Best Model): {mse_best}')
print(f'Root Mean Squared Error (Best Model): {rmse_best}')
print(f'R-squared (R2) Score (Best Model): {r2_best}')

# Now you can use the best model for predictions
# For example, you can predict the yield for a new set of features:
new_data = pd.DataFrame({'Weather': [25], 'Soil_Type': [1], 'Fertilizer': [200], 'Planting_Density': [150]})
predicted_yield_new = best_rf_model.predict(new_data)
print(f'Predicted Yield (Best Model): {predicted_yield_new[0]}')


# In[ ]:





# In[ ]:





# In[4]:


#import libraries for access and functional purpose
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as s
df_input=pd.read_csv(r"C:\Users\hp\Desktop\\heart.csv")
df_input.shape
df_input.head()
df_input['Crop'].nunique()
df_input['Crop'].value_counts()
df_input['District_Name'].nunique()
df_input['District_Name'].value_counts()
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
categorical_columns = ['District_Name', 'Crop' ,'Season']
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
def trained(clf,x_train,x_test,y_train,y_test):
    clf.fit(x_train,y_train)
    y_train_pred = clf.predict(x_train)
    29
    print("MSE " + str(mean_squared_error(y_train, y_train_pred)))
    print("MAE " + str(mean_absolute_error(y_train, y_train_pred)))
    print("RMSE " + str(np.sqrt(mean_squared_error(y_train, y_train_pred))))
#print("RMSLE " + str(np.sqrt(mean_squared_log_error(y_test, y_pred))))
R2_score = r2_score(y_train, y_train_pred)
print("R2 Score " + str(R2_score))
y_pred = np.array(y_train_pred)
y_test = np.array(y_train)
data={}
data['x'] = y_test
data['y'] = y_pred
sns.regplot(x="x", y="y", data=data);
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
#print("RMSLE " + str(np.sqrt(mean_squared_log_error(y_test, y_pred))))
R2_score = r2_score(y_test,y_pred)
print("R2 Score " + str(R2_score))
y_pred = np.array(y_pred)
y_test = np.array(y_test)
data={}
data['x'] = y_test
data['y'] = y_pred
sns.regplot(x="x", y="y", data=data);


# In[ ]:




