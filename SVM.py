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

#GUI START
from tkinter import *
root=Tk()
root.configure(background='#E8EDF4')
root.title("Heart Disease Prediction Using Machine Learning")

#Heading
w2=Label(root,justify=LEFT,text="Heart Disease Predictor",font='times 24 bold italic',fg="blue",bg="#E8EDF4")
w2.grid(row=1,column=0,columnspan=4,padx=100,pady=20)

#Patient Name
Name=Label(root,text="Patient Name",bg="#E8EDF4")
Name.grid(row=2,column=0,pady=15)
Name=Entry(root,textvariable=StringVar())
Name.grid(row=2,column=1,pady=10)

#Age
Age=Label(root,text="Age",bg="#E8EDF4")
Age.grid(row=3,column=0,pady=10)
age=Entry(root,textvariable=IntVar())
age.grid(row=3,column=1,pady=10)

#Sex
Sex=Label(root,text="Sex",bg="#E8EDF4")
Sex.grid(row=4,column=0,pady=10)
sex=StringVar()
sex.set("Male")
sex_ans={"Male":1,"Female":0}
drop=OptionMenu(root,sex,*sex_ans.keys())
drop.config(bg="#E8EDF4")
drop.grid(row=4,column=1,pady=10)

#Education
edu=Label(root,text="Education",bg="#E8EDF4")
edu.grid(row=5,column=0,pady=10)
education = StringVar()
education.set("10th")
education_levels={"10th":1,"12th":2,"College":3,"Graduate":4}
drop=OptionMenu(root,education,*education_levels.keys())
drop.config(bg="#E8EDF4")
drop.grid(row=5,column=1,pady=10)

#Smoking
Smoking=Label(root,text="Smoking",bg="#E8EDF4")
Smoking.grid(row=6,column=0,pady=10)
smoking=StringVar()
smoking.set("Yes")
smoking_ans={"Yes":1,"No":0}
drop=OptionMenu(root,smoking,*smoking_ans.keys())
drop.config(bg="#E8EDF4")
drop.grid(row=6,column=1,pady=10)

#No. of cigarettes
Nocig=Label(root,text="No. of Cigarettes per day",bg="#E8EDF4")
Nocig.grid(row=7,column=0,pady=10)
Cigratte=Entry(root,textvariable=IntVar())
Cigratte.grid(row=7,column=1,pady=10)

#BP Medicines
BPmed=Label(root,text="B.P Medicine",bg="#E8EDF4")
BPmed.grid(row=8,column=0,pady=10)
Medicine = StringVar()
Medicine.set("Yes")
Medicine_ans={"Yes":1,"No":0}
drop=OptionMenu(root,Medicine,*Medicine_ans.keys())
drop.config(bg="#E8EDF4")
drop.grid(row=8,column=1,pady=10)

#Prevalent Stroke
PrevalentStroke=Label(root,text="Prevalent Stroke",bg="#E8EDF4")
PrevalentStroke.grid(row=9,column=0,pady=10)
Stroke=StringVar()
Stroke.set("Yes")
Stroke_ans={"Yes":1,"No":0}
drop=OptionMenu(root,Stroke,*Stroke_ans.keys())
drop.config(bg="#E8EDF4")
drop.grid(row=9,column=1,pady=10)

#Diabetes
diabetes=Label(root,text="Diabetes",bg="#E8EDF4")
diabetes.grid(row=2,column=2,pady=10)
Diabetes = StringVar()
Diabetes.set("Yes")
Diabetes_ans={"Yes":1,"No":0}
drop=OptionMenu(root,Diabetes,*Diabetes_ans.keys())
drop.config(bg="#E8EDF4")
drop.grid(row=2,column=3,pady=10)

#TotChol
totChol=Label(root,text="TotChol",bg="#E8EDF4")
totChol.grid(row=3,column=2,pady=10)
totChol=Entry(root,textvariable=IntVar())
totChol.grid(row=3,column=3,pady=10)

#BMI
BMI=Label(root,text="BMI",bg="#E8EDF4")
BMI.grid(row=4,column=2,pady=10)
BMI=Entry(root,textvariable=IntVar())
BMI.grid(row=4,column=3,pady=10)

#HeartRate
heartrate=Label(root,text="Heart Rate",bg="#E8EDF4")
heartrate.grid(row=5,column=2,pady=10)
heartrate=Entry(root,textvariable=IntVar())
heartrate.grid(row=5,column=3,pady=10)

#Gulcose
glucose=Label(root,text="Glucose",bg="#E8EDF4")
glucose.grid(row=6,column=2,pady=10)
glucose=Entry(root,textvariable=IntVar())
glucose.grid(row=6,column=3,pady=10)

#SysBp
sysBP=Label(root,text="Systolic B.P",bg="#E8EDF4")
sysBP.grid(row=7,column=2,pady=10)
sysBP=Entry(root,textvariable=IntVar())
sysBP.grid(row=7,column=3,pady=10)

#DiaBp
diaBP=Label(root,text="Diastolic B.P",bg="#E8EDF4")
diaBP.grid(row=8,column=2,pady=10)
diaBP=Entry(root,textvariable=IntVar())
diaBP.grid(row=8,column=3,pady=10)

#Prevalent Hyp
PrevalentHyp=Label(root,text="Prevalent Hyp",bg="#E8EDF4")
PrevalentHyp.grid(row=9,column=2,pady=10)
Hyp=StringVar()
Hyp.set("Yes")
Hyp_ans={"Yes":1,"No":0}
drop=OptionMenu(root,Hyp,*Hyp_ans.keys())
drop.config(bg="#E8EDF4")
drop.grid(row=9,column=3,pady=10)

#Predict
def myclick():
    result=""
    predict = svc.predict([np.asarray((sex_ans[sex.get()], age.get(), Cigratte.get(), Medicine_ans[Medicine.get()],
                                      Stroke_ans[Stroke.get()], Hyp_ans[Hyp.get()], Diabetes_ans[Diabetes.get()],
                                      totChol.get(),
                                      sysBP.get(), diaBP.get(), BMI.get(), heartrate.get(), glucose.get()),
                                     dtype='float64')])
    print(predict[0])
    if(predict[0]==0):
        result="Not Diseased"
    else:
        result="Diseased"
    output.config(text=result)

predict=Button(root,text="Predict",command=myclick)
predict.config(height=2,width=20,bg="#CAD6E6")
predict.grid(row=10,column=1,pady=40,columnspan=2)

output = Label(root, text="", bg="#E8EDF4")
output.grid(row=11,column=1)

root.resizable(width=False,height=False)
root.mainloop()

#End of GUI BACKEND
