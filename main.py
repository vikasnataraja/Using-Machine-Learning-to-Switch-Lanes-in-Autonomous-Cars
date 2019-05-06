#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier,ExtraTreeClassifier
from sklearn.svm import SVC


# ## Read in X and y using pandas

# In[ ]:


columnsList = ['alternate_drive', 'bike_angle', 'bike_area', 'bike_dist', 'bus_angle', 'bus_area', 
               'bus_dist', 'car_angle', 'car_area', 'car_dist', 'direct_drive', 'left_lane_direction', 
               'left_lane_style', 'left_lane_type', 'motor_angle', 'motor_area', 'motor_dist', 'person_angle', 
               'person_area', 'person_dist', 'rider_angle', 'rider_area', 'rider_dist', 'right_lane_direction', 
               'right_lane_style', 'right_lane_type', 'scene', 'time', 'traff_lights_angle', 'traff_lights_area', 
               'traff_lights_color', 'traff_lights_dist', 'traff_sign_angle', 'traff_sign_area', 'traff_sign_dist',
               'train_angle', 'train_area', 'train_dist', 'truck_angle', 'truck_area', 'truck_dist', 'weather']

X = pd.read_csv("X",header=None)
X.columns=columnsList

y = pd.read_csv("y",header=None)
y.columns = ["outcome"]


# In[ ]:


X = X.astype('float64')
y = y.astype('int64')
y=y.iloc[:,0]


# ## Display the data

# In[ ]:


X


# In[ ]:


y


# ## Fit and predict 
# 
# * Use cross validation to find the best fit
# * Here, I'm only showing 3 models. In reality, I used over 10 models but found these to be the best ones

# In[ ]:


#for i in range(1,33,1):

modelGRAD = GradientBoostingClassifier(learning_rate=0.12,n_estimators=100,min_samples_split=2,
                                       random_state=42,max_features=10,max_depth=2)
scores = cross_val_score(modelGRAD, X, y, cv=5)
#print(i)
print(scores)
print(sum(scores))
print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# In[ ]:



clf1 = RandomForestClassifier(n_estimators=90,random_state=42,max_features=20,bootstrap=True,max_depth=18,
                              oob_score=True, min_samples_split=10,criterion='gini')
scores = cross_val_score(clf1, X, y, cv=5)
print(i)
print(scores)
print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# In[ ]:



modelada = AdaBoostClassifier(learning_rate=0.40,random_state=42,n_estimators=50)
scores = cross_val_score(modelada, X, y, cv=5)
print(i)
print(scores)
print(sum(scores))
print("Accuracy: %0.4f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# ## Visualize the results
# 
# 

# In[ ]:


co = y.value_counts()
values = [co[0],co[1]]
names = ['Do not switch lanes', 'Can switch lane']
plt.bar(names, values)
plt.title('Distribution of outcome (y) variable')
plt.savefig('y_distribution.png')


# In[ ]:


objects = ('Random Forest', 'Logistic Regression', 'AdaBoost', 'Gradient Boosting','Bagging Classifier')
y_pos = np.arange(len(objects))
performance = [82.38,81.91,82.61,83.00,81.12]

plt.figure(figsize=(10,5))
plt.bar(y_pos, performance, align='center')
plt.xticks(y_pos, objects)
plt.ylabel('Accuracy in %')
plt.ylim(80,85)

plt.title('Comparison of accuracies')

#plt.show()
plt.savefig('acc_algo.png')


# In[ ]:




