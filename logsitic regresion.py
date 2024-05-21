import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, confusion_matrix, r2_score, accuracy_score
from sklearn.model_selection import (GridSearchCV, KFold, train_test_split, cross_val_score)

from imblearn.over_sampling import SMOTE
from collections import Counter

from sklearn.tree import DecisionTreeClassifier
# from catboost import CatBoostClassifier

# """ Import the dataset """"

path ="C:/Users/joy98/OneDrive/Desktop/depeesh sir train data model/water_potability (1).csv"
df = pd.read_csv(path)

# Visualize the dataset 
df.info()

# Visualization of the datadistribution 

df.describe().T.style.background_gradient(subset=['mean','std','50%','count'], cmap='PuBu')

# Check for missing values
plt.title('Missing Values Per Feature')
nans = df.isna().sum().sort_values(ascending=False).to_frame()
sns.heatmap(nans,annot=True,fmt='d',cmap='vlag')


# Imputing the missing values with the mean
#################################### Imputing 'ph' value #####################################

phMean_0 = df[df['Potability'] == 0]['ph'].mean(skipna=True)
df.loc[(df['Potability'] == 0) & (df['ph'].isna()), 'ph'] = phMean_0
phMean_1 = df[df['Potability'] == 1]['ph'].mean(skipna=True)
df.loc[(df['Potability'] == 1) & (df['ph'].isna()), 'ph'] = phMean_1

##################################### Imputing 'Sulfate' value #####################################

SulfateMean_0 = df[df['Potability'] == 0]['Sulfate'].mean(skipna=True)
df.loc[(df['Potability'] == 0) & (df['Sulfate'].isna()), 'Sulfate'] = SulfateMean_0
SulfateMean_1 = df[df['Potability'] == 1]['Sulfate'].mean(skipna=True)
df.loc[(df['Potability'] == 1) & (df['Sulfate'].isna()), 'Sulfate'] = SulfateMean_1

################################ Imputing 'Trihalomethanes' value #####################################

TrihalomethanesMean_0 = df[df['Potability'] == 0]['Trihalomethanes'].mean(skipna=True)
df.loc[(df['Potability'] == 0) & (df['Trihalomethanes'].isna()), 'Trihalomethanes'] = TrihalomethanesMean_0
TrihalomethanesMean_1 = df[df['Potability'] == 1]['Trihalomethanes'].mean(skipna=True)
df.loc[(df['Potability'] == 1) & (df['Trihalomethanes'].isna()), 'Trihalomethanes'] = TrihalomethanesMean_1

################################ See if imputer has worked #####################################

print('Checking to see any more missing data \n')
df.isna().sum()

################################     Correlation plot       #####################################
Corrmat = df.corr()
plt.subplots(figsize=(7,7))
sns.heatmap(Corrmat, cmap="YlGnBu", square = True, annot=True, fmt='.2f')
plt.show()


##################### Preparing the Data for Modelling  ######################

X = df.drop('Potability', axis = 1).copy()
y = df['Potability'].copy()

############################# Train-Test split ############################
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25)

########################## Synthetic OverSampling ###########################
print('Balancing the data by SMOTE - Oversampling of Minority level\n')
smt = SMOTE()
counter = Counter(y_train)
print('Before SMOTE', counter)
X_train, y_train = smt.fit_resample(X_train, y_train)
counter = Counter(y_train)
print('\nAfter SMOTE', counter)

################################# Scaling #################################
ssc = StandardScaler()

X_train = ssc.fit_transform(X_train)
X_test = ssc.transform(X_test)

modelAccuracy = list()


#################################### LogisticRegression Classifier() #######################
from sklearn.linear_model import LogisticRegression
print('LogisticRegression Classifier\n')
logistic_regressor = LogisticRegression()
logistic_regressor.fit(X_train, y_train)
y_predict = logistic_regressor.predict(X_test)
print(metrics.classification_report(y_test, y_predict))
print(modelAccuracy.append(metrics.accuracy_score(y_test, y_predict)))

#sns.heatmap(confusion_matrix(y_test, y_predict), annot=True, fmt='d')
#plt.show()
########################confusion matrix of catboost############################
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
result = confusion_matrix(y_test, y_predict)
print("logistic Confusion Matrix:")
print(result)
f, ax=plt.subplots(figsize=(8,5))
sns.heatmap(result/np.sum(result), annot = True, fmt=  '0.2%', cmap = 'Reds')
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.title('Confusion Matrix')
plt.show()

