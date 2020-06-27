#Import libraries
import pandas as pd
import numpy as np
import csv
import matplotlib.pylab as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tkinter import *

#Importing Data Set in df from HeartDiseasePrediction.csv
filename="C:\\Users\\win 10\\B.tech\\2nd Year\\HeartDiseasePrediction.csv"
df=pd.read_csv(filename)

#According to Data Analysis, Droping Current Smoker and Education Column
df.drop(['currentSmoker'],axis=1,inplace=True)
df.drop(['education'],axis=1,inplace=True)
df.dropna(axis=0,inplace=True)

#Selecting Independent Factors for Determining Heart Disease
X = df[['male', 'age','cigsPerDay', 'BPMeds', 'prevalentStroke', 'prevalentHyp','diabetes', 'totChol', 'sysBP','diaBP','BMI','heartRate','glucose']] .values
X[0:5]
y = df['TenYearCHD'].values

#Spilting Data Set
#20% Dataset for Testing Purpose And 80% Dataset for Training Purpose.

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2,random_state=5)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
#Printing the Shape of Training Dataset And Testing Dataset
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

#Applying Logistic Algorithm For Prediction
LR = LogisticRegression(C=0.01,solver='liblinear').fit(X_train,y_train)
#Predict Testing Data
predTest= LR.predict(X_test)

#Accuracy of Logistic Regression
print("Test set Accuracy: ", metrics.accuracy_score(y_test, predTest))

results=confusion_matrix(y_test,predTest)
print(results)
'''
     y=1/(1+e^(-z))
     where,
      y is the output of logistic regression model for a particular example.
      z = b + w1*x1 + w2*x2 + w3*x3 + w4*x4  ........ + wn*xn
      The w values are the model's learned weights, and b is the bias.
      The x values are the feature values for a particular example
      z can also be represented like this z=log(y/(1-y))
'''
#START Making GUI
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

#Gender
Gender=Label(root,text="Gender",bg="#E8EDF4")
Gender.grid(row=4,column=0,pady=10)
gender=StringVar()
gender.set("Male")
gender_ans={"Male":1,"Female":0}
drop=OptionMenu(root,gender,*gender_ans.keys())
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
BMI=Entry(root,textvariable=DoubleVar())
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
sysBP=Entry(root,textvariable=DoubleVar())
sysBP.grid(row=7,column=3,pady=10)

#DiaBp
diaBP=Label(root,text="Diastolic B.P",bg="#E8EDF4")
diaBP.grid(row=8,column=2,pady=10)
diaBP=Entry(root,textvariable=DoubleVar())
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

#GUI Backend
#Predicting Function
def myclick():
    result=""
    predict=LR.predict([np.asarray((gender_ans[gender.get()],age.get(),Cigratte.get(),Medicine_ans[Medicine.get()],
                            Stroke_ans[Stroke.get()],Hyp_ans[Hyp.get()],Diabetes_ans[Diabetes.get()],totChol.get(),
                                  sysBP.get(),diaBP.get(),BMI.get(),heartrate.get(),glucose.get()),dtype='float64')])
    #Determing Diseased or Not Diseased
    if(predict[0]==0):
        result="Not Diseased"
        output.config(text=result, foreground="green", font='times 20 bold italic')
    else:
        result="Diseased"
        output.config(text=result, foreground="red", font='times 20 bold italic')
    #Set Output in output label

predict=Button(root,text="Predict",command=myclick)
predict.config(height=2,width=20,bg="#CAD6E6")
predict.grid(row=10,column=1,pady=40,columnspan=2)

output = Label(root, text="", bg="#E8EDF4")

output.grid(row=11,column=1)

root.resizable(width=False,height=False)
root.mainloop()

#End of GUI BACKEND

