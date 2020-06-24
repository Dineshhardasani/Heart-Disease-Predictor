import pandas as pd
import numpy as np
import csv
import matplotlib.pylab as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC


# import metrics to compute accuracy
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


filename="HeartDiseasePrediction.csv"
df=pd.read_csv(filename)
df.drop(['currentSmoker'],axis=1,inplace=True)
df.drop(['education'],axis=1,inplace=True)
df.dropna(axis=0,inplace=True)

X = df[['male', 'age','cigsPerDay', 'BPMeds', 'prevalentStroke', 'prevalentHyp','diabetes', 'totChol', 'sysBP','diaBP','BMI','heartRate','glucose']] .values  #.astype(float)

y = df['TenYearCHD'].values

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2,random_state=5)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)
print(X_train)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

print(X_train)


#X_train.head()

# instantiate classifier
svc=SVC(C=1.0,kernel='rbf',gamma=2.0)


# fit classifier to training set
svc.fit(X_train,y_train)


y_train_pred = svc.predict(X_train)
print('Model accuracy score for train data: {}'. format(accuracy_score(y_train, y_train_pred)))
# make predictions on test set
y_pred=svc.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print('Model accuracy score for test data: {}'. format(accuracy_score(y_test, y_pred)))

#Finding good hyperparameters for the model using GridSearchCV
'''C_range = np.logspace(0.01, 10, 12)
gamma_range = np.logspace(0.01, 3, 12)
parameters = dict(gamma=gamma_range, C=C_range)

grid_search = GridSearchCV(estimator = svc,
                           param_grid = parameters,
                           cv = 10,
                           scoring='accuracy')
grid_search = grid_search.fit(X_train, y_train)

accuracy = grid_search.best_score_
param = grid_search.best_params_'''
