from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
import seaborn as sns
import pydotplus
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

######## Preprocessing ###########
df = pd.read_csv("./data/CustomersDataset.csv")

df['TotalCharges'] = df["TotalCharges"].replace(" ", np.nan)
df.dropna()
df["TotalCharges"] = df["TotalCharges"].astype(float)


def replace_gender_value(value):
    if value == "Male":
        return 0
    else:
        return 1


df["gender"] = df["gender"].apply(replace_gender_value, 1)


def replace_Partner_value(value):
    if value == "No":
        return 0
    else:
        return 1


df["Partner"] = df["Partner"].apply(replace_Partner_value, 1)
df["Dependents"].replace({
    "No": 0,
    "Yes": 1
}, inplace=True)
df["PhoneService"].replace({
    "No": 0,
    "Yes": 1
}, inplace=True)
df["MultipleLines"].replace({
    "No": 0,
    "No phone service": 0,
    "Yes": 1
}, inplace=True)
df["InternetService"].replace({
    "No": 0,
    "DSL": 1,
    "Fiber optic": 2
}, inplace=True)
df["OnlineSecurity"].replace({
    "No": 0,
    "No internet service": 0,
    "Yes": 1
}, inplace=True)
df["OnlineBackup"].replace({
    "No": 0,
    "No internet service": 0,
    "Yes": 1
}, inplace=True)
df["DeviceProtection"].replace({
    "No": 0,
    "No internet service": 0,
    "Yes": 1
}, inplace=True)
df["TechSupport"].replace({
    "No": 0,
    "No internet service": 0,
    "Yes": 1
}, inplace=True)
df["StreamingTV"].replace({
    "No": 0,
    "No internet service": 0,
    "Yes": 1
}, inplace=True)
df["StreamingMovies"].replace({
    "No": 0,
    "No internet service": 0,
    "Yes": 1
}, inplace=True)
df["Contract"].replace({
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
}, inplace=True)
df["PaperlessBilling"].replace({
    "No": 0,
    "Yes": 1
}, inplace=True)
df["PaymentMethod"].replace({
    "Electronic check": 0,
    "Mailed check": 1,
    "Bank transfer (automatic)": 2,
    "Credit card (automatic)": 3
}, inplace=True)
df["Churn"].replace({
    "No": 0,
    "Yes": 1
}, inplace=True)
MinMaxScaler = MinMaxScaler()
columns_for_scaling = ['tenure', 'MonthlyCharges', 'TotalCharges']
df[columns_for_scaling] = MinMaxScaler.fit_transform(df[columns_for_scaling])
df.drop("customerID", axis=1, inplace=True)
x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values
df.dropna(inplace=True)
imputer = SimpleImputer(missing_values=np.nan)
imputer.fit(x[:, :19])
df.to_csv("new_df.csv", index=False)
new_dafr = pd.read_csv("new_df.csv")

y = df['Churn']
X = df.drop(['Churn'], axis=1)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=50)

####### Regression Logistic#######

from sklearn.linear_model import LogisticRegression
from sklearn import metrics

logmodel = LogisticRegression(random_state=50)
logmodel.fit(X_train.values, y_train.values)
LogPredict = logmodel.predict(X_test.values)
logmodelAccuracy = metrics.accuracy_score(y_test, LogPredict)
############  SVM  ###############
from sklearn.svm import SVC

svcmodel = SVC(kernel='linear', random_state=50, probability=True)
svcmodel.fit(X_train.values, y_train.values)
svcPredict = svcmodel.predict(X_test.values)
svcAccuracy = metrics.accuracy_score(y_test, svcPredict)
################ DecisionTree ###############
from sklearn.tree import DecisionTreeClassifier

dtmodel = DecisionTreeClassifier(criterion="entropy", random_state=50)
dtmodel.fit(X_train.values, y_train.values)
dtPredict = dtmodel.predict(X_test.values)
dtAccuracy = metrics.accuracy_score(y_test, dtPredict)
############# KNN ###############
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

knnModel = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
knnModel.fit(X_train.values, y_train.values)
knnPredict = knnModel.predict(X_test.values)
knnAccuracy = metrics.accuracy_score(y_test, knnPredict)

bayesModel = GaussianNB()
bayesModel.fit(X_train, y_train)
bayesPredict = bayesModel.predict(X_test)
bayesAccuracy = metrics.accuracy_score(y_test, bayesPredict)


##########  FUN  #################


def TestLR() -> object:
    print("Accuracy of Logistic Regression (Test): ")
    print(logmodelAccuracy)


def TrainLR():
    print("Accuracy of Logistic Regression (Train): ")
    print(logmodel.score(X_train, y_train))


def TestSVM():
    print("Accuracy of SVM (Test): ")
    print(svcAccuracy)


def TrainSVM():
    print("Accuracy of SVM (Train): ")
    print(svcmodel.score(X_train, y_train))


def TestID3():
    print("Accuracy of Decision Tree (Test): ")
    print(dtAccuracy)


def TrainID3():
    print("Accuracy of ID3 (Train): ")
    print(dtmodel.score(X_train, y_train))


def TestKNN():
    print("Accuracy of KNN (Test): ")
    print(knnAccuracy)


def TrainKNN():
    print("Accuracy of KNN(Train): ")
    print(knnModel.score(X_train, y_train))


def TestBayes():
    print("Accuracy of Naive Bayes(Test): ")
    print(bayesAccuracy)


def TrainBayes():
    print("Accuracy of Naive Bayes(Train): ")
    print(knnModel.score(X_train, y_train))


def test():
    if (var1.get() == 1 and var2.get() == 1 and var3.get() == 1 and var4.get() == 1 and var5.get() == 1):
        TestLR()
        TestSVM()
        TestID3()
        TestKNN()
        TestBayes()
    elif (var1.get() == 1 and var2.get() == 1 and var3.get() == 1 and var4.get() == 1):
        TestLR()
        TestSVM()
        TestID3()
        TestKNN()
    elif (var1.get() == 1 and var2.get() == 1 and var3.get() == 1 and var5.get() == 1):
        TestLR()
        TestSVM()
        TestID3()
        TestBayes()
    elif (var1.get() == 1 and var2.get() == 1 and var4.get() == 1 and var5.get() == 1):
        TestLR()
        TestSVM()
        TestKNN()
        TestBayes()
    elif (var1.get() == 1 and var3.get() == 1 and var4.get() == 1 and var5.get() == 1):
        TestLR()
        TestID3()
        TestKNN()
        TestBayes()
    elif (var2.get() == 1 and var3.get() == 1 and var4.get() == 1 and var5.get() == 1):
        TestSVM()
        TestID3()
        TestKNN()
        TestBayes()
    elif (var1.get() == 1 and var2.get() == 1 and var3.get() == 1):
        TestLR()
        TestSVM()
        TestID3()
    elif (var1.get() == 1 and var2.get() == 1 and var4.get() == 1):
        TestLR()
        TestSVM()
        TestKNN()
    elif (var1.get() == 1 and var2.get() == 1 and var5.get() == 1):
        TestLR()
        TestSVM()
        TestBayes()
    elif (var1.get() == 1 and var3.get() == 1 and var4.get() == 1):
        TestLR()
        TestID3()
        TestKNN()
    elif (var1.get() == 1 and var3.get() == 1 and var5.get() == 1):
        TestLR()
        TestID3()
        TestBayes()
    elif (var2.get() == 1 and var3.get() == 1 and var4.get() == 1):
        TestSVM()
        TestID3()
        TestKNN()
    elif (var2.get() == 1 and var3.get() == 1 and var5.get() == 1):
        TestSVM()
        TestID3()
        TestBayes()
    elif (var2.get() == 1 and var4.get() == 1 and var5.get() == 1):
        TestSVM()
        TestKNN()
        TestBayes()
    elif (var1.get() == 1 and var4.get() == 1 and var5.get() == 1):
        TestLR()
        TestKNN()
        TestBayes()
    elif (var1.get() == 1 and var2.get() == 1):
        TestLR()
        TestSVM()
    elif (var1.get() == 1 and var3.get() == 1):
        TestLR()
        TestID3()
    elif (var1.get() == 1 and var4.get() == 1):
        TestLR()
        TestKNN()
    elif (var1.get() == 1 and var5.get() == 1):
        TestLR()
        TestBayes()
    elif (var2.get() == 1 and var3.get() == 1):
        TestSVM()
        TestID3()
    elif (var2.get() == 1 and var4.get() == 1):
        TestSVM()
        TestKNN()
    elif (var2.get() == 1 and var5.get() == 1):
        TestSVM()
        TestBayes()
    elif (var3.get() == 1 and var4.get() == 1):
        TestID3()
        TestKNN()
    elif (var3.get() == 1 and var5.get() == 1):
        TestID3()
        TestBayes()
    elif (var1.get() == 1):
        TestLR()
    elif (var2.get() == 1):
        TestSVM()
    elif (var3.get() == 1):
        TestID3()
    elif (var4.get() == 1):
        TestKNN()
    elif (var5.get() == 1):
        TestBayes()


def train():
    if (var1.get() == 1 and var2.get() == 1 and var3.get() == 1 and var4.get() == 1 and var5.get() == 1):
        TrainLR()
        TrainSVM()
        TrainID3()
        TrainKNN()
        TrainBayes()

    elif (var1.get() == 1 and var2.get() == 1 and var3.get() == 1 and var4.get() == 1):
        TrainLR()
        TrainSVM()
        TrainID3()
        TrainKNN()
    elif (var1.get() == 1 and var2.get() == 1 and var3.get() == 1 and var5.get() == 1):
        TrainLR()
        TrainSVM()
        TrainID3()
        TrainBayes()
    elif (var1.get() == 1 and var2.get() == 1 and var4.get() == 1 and var5.get() == 1):
        TrainLR()
        TrainSVM()
        TrainKNN()
        TrainBayes()
    elif (var1.get() == 1 and var3.get() == 1 and var4.get() == 1 and var5.get() == 1):
        TrainLR()
        TrainID3()
        TrainKNN()
        TrainBayes()
    elif (var2.get() == 1 and var3.get() == 1 and var4.get() == 1 and var5.get() == 1):
        TrainSVM()
        TrainID3()
        TrainKNN()
        TrainBayes()
    elif (var1.get() == 1 and var2.get() == 1 and var3.get() == 1):
        TrainLR()
        TrainSVM()
        TrainID3()
    elif (var1.get() == 1 and var2.get() == 1 and var4.get() == 1):
        TrainLR()
        TrainSVM()
        TrainKNN()
    elif (var1.get() == 1 and var2.get() == 1 and var5.get() == 1):

        TrainLR()
        TrainSVM()
        TrainBayes()

    elif (var1.get() == 1 and var3.get() == 1 and var4.get() == 1):
        TrainLR()
        TrainID3()
        TrainKNN()
    elif (var1.get() == 1 and var3.get() == 1 and var5.get() == 1):

        TrainLR()
        TrainID3()
        TrainBayes()
    elif (var2.get() == 1 and var3.get() == 1 and var4.get() == 1):
        TrainSVM()
        TrainID3()
        TrainKNN()
    elif (var2.get() == 1 and var3.get() == 1 and var5.get() == 1):
        TrainSVM()
        TrainID3()
        TrainBayes()

    elif (var2.get() == 1 and var4.get() == 1 and var5.get() == 1):
        TrainSVM()
        TrainKNN()
        TrainBayes()
    elif (var1.get() == 1 and var4.get() == 1 and var5.get() == 1):
        TrainLR()
        TrainKNN()
        TrainBayes()
    elif (var1.get() == 1 and var2.get() == 1):

        TrainLR()
        TrainSVM()

    elif (var1.get() == 1 and var3.get() == 1):

        TrainLR()
        TrainID3()

    elif (var1.get() == 1 and var4.get() == 1):

        TrainLR()
        TrainKNN()
    elif (var1.get() == 1 and var5.get() == 1):

        TrainLR()
        TrainBayes()

    elif (var2.get() == 1 and var3.get() == 1):
        TrainSVM()
        TrainID3()

    elif (var2.get() == 1 and var4.get() == 1):
        TrainSVM()
        TrainKNN()
    elif (var2.get() == 1 and var5.get() == 1):
        TrainSVM()
        TrainBayes()

    elif (var3.get() == 1 and var4.get() == 1):
        TrainID3()
        TrainKNN()

    elif (var3.get() == 1 and var5.get() == 1):

        TrainID3()
        TrainBayes()
    elif (var1.get() == 1):
        TrainLR()
    elif (var2.get() == 1):
        TrainSVM()
    elif (var3.get() == 1):
        TrainID3()
    elif (var4.get() == 1):
        TrainKNN()
    elif (var5.get() == 1):
        TrainBayes()


def dataframes():
    en1 = ent9.get()
    if (en1 == "Yes" or en1 == "yes"):
        en1 = 1
    elif (en1 == "No" or en1 == "no"):
        en1 = 0
    Label(view, text=en1)
    en2 = ent3.get()
    if (en2 == "Yes" or en2 == "yes"):
        en2 = 1
    elif (en2 == "No" or en2 == "no"):
        en2 = 0
    Label(view, text=en2)
    en3 = ent10.get()
    if (en3 == "Yes" or en3 == "yes"):
        en3 = 1
    elif (en3 == "No" or en3 == "no" or en3 == 'No phone service'):
        en3 = 0
    Label(view, text=en3)
    en4 = ent17.get()
    if (en4 == "DSL" or en4 == "dsl"):
        en4 = 1
    elif (en4 == "No" or en4 == "no"):
        en4 = 0
    elif (en4 == "Fiber optic" or en4 == "fiber optic"):
        en4 = 2
    Label(view, text=en4)
    en5 = ent4.get()
    if (en5 == "Yes" or en5 == "yes"):
        en5 = 1
    elif (en5 == "No" or en5 == "no" or en5 == "No internet service"):
        en5 = 0
    Label(view, text=en5)
    en6 = ent11.get()
    if (en6 == "Yes" or en6 == "yes"):
        en6 = 1
    elif (en6 == "No" or en6 == "no" or en6 == "No internet service"):
        en6 = 0
    Label(view, text=en6)
    en7 = ent18.get()
    if (en7 == "Yes" or en7 == "yes"):
        en7 = 1
    elif (en7 == "No" or en7 == "no" or en7 == "No internet service"):
        en7 = 0
    Label(view, text=en7)
    en8 = ent5.get()
    if (en8 == "Yes" or en8 == "yes"):
        en8 = 1
    elif (en8 == "No" or en8 == "no" or en8 == "No internet service"):
        en8 = 0
    Label(view, text=en8)
    en9 = ent12.get()
    if (en9 == "Yes" or en9 == "yes"):
        en9 = 1
    elif (en9 == "No" or en9 == "no" or en9 == "No internet service"):
        en9 = 0
    Label(view, text=en9)
    en10 = ent19.get()
    if (en10 == "Yes" or en10 == "yes"):
        en10 = 1
    elif (en10 == "No" or en10 == "no" or en10 == "No internet service"):
        en10 = 0
    Label(view, text=en10)
    en11 = ent6.get()
    if (en11 == "One year" or en11 == "one year"):
        en11 = 1
    elif (en11 == "Month-to-month" or en11 == "month-to-month"):
        en11 = 0
    elif (en11 == "Two year" or en11 == "two year"):
        en11 = 2
    Label(view, text=en11)
    en12 = ent13.get()
    if (en12 == "Yes" or en12 == "yes"):
        en12 = 1
    elif (en12 == "No" or en12 == "no"):
        en12 = 0
    Label(view, text=en12)
    en13 = ent20.get()
    if (en13 == "Mailed check" or en13 == "mailed check"):
        en13 = 1
    elif (en13 == "Electronic check" or en13 == "electronic check"):
        en13 = 0
    elif (en13 == "Bank transfer" or en13 == "bank transfer"):
        en13 = 2
    elif (en13 == "Credit card" or en13 == "credit card"):
        en13 = 3
    Label(view, text=en13)
    en14 = ent8.get()
    if (en14 == "Female" or en14 == "female"):
        en14 = 1
    elif (en14 == "Male" or en14 == "male"):
        en14 = 0
    Label(view, text=en14)
    en15 = ent2.get()
    if (en15 == "yes" or en15 == "Yes"):
        en15 = 1
    elif (en15 == "No" or en15 == "no"):
        en15 = 0
    Label(view, text=en15)
    en16 = ent16.get()
    if (en16 == "yes" or en16 == "Yes"):
        en16 = 1
    elif (en16 == "No" or en16 == "no"):
        en16 = 0
    Label(view, text=en16)
    en17 = ent7.get()
    Label(view, text=en17)
    en18 = ent14.get()
    Label(view, text=en18)
    en19 = ent15.get()
    Label(view, text=en19)

    arr = [[en1, en2, en3, en4, en5, en6, en7, en8, en9, en10, en11, en12, en13, en14, en15, en16, en17, en18, en19]]

    def predictLR():

        print('the prediction by Logistic Regression is:')
        y = logmodel.predict(arr)
        if (y == 1):
            print("Yes")
        elif (y == 0):
            print("No")

    def predictSVM():
        print('the prediction by SVM is:')
        y = svcmodel.predict(arr)
        if (y == 1):
            print("Yes")
        elif (y == 0):
            print("No")

    def predictID3():
        print('the prediction by id3 is:')
        y = dtmodel.predict(arr)
        if (y == 0):
            print("Yes")
        elif (y == 1):
            print('No')

    def predknn():
        print('the prediction by  K Nearest Neighbor Classifier is:')
        y = knnModel.predict(arr)
        if (y == 1):
            print("Yes")
        elif (y == 0):
            print("No")

    def predbayes():
        print('the prediction by Naive Bayes classifier is:')
        y = bayesModel.predict(arr)
        if (y == 1):
            print("Yes")
        elif (y == 0):
            print("No")

    if (var1.get() == 1 and var2.get() == 1 and var3.get() == 1 and var4.get() == 1 and var5.get() == 1):
        predictLR()
        predictSVM()
        predictID3()
        predknn()
        predbayes()
    elif (var1.get() == 1 and var2.get() == 1 and var3.get() == 1 and var4.get() == 1):
        predictLR()
        predictSVM()
        predictID3()
        predknn()
    elif (var1.get() == 1 and var2.get() == 1 and var3.get() == 1 and var5.get() == 1):
        predictLR()
        predictSVM()
        predictID3()
        predbayes()
    elif (var1.get() == 1 and var2.get() == 1 and var4.get() == 1 and var5.get() == 1):
        predictLR()
        predictSVM()
        predknn()
        predbayes()
    elif (var1.get() == 1 and var3.get() == 1 and var4.get() == 1 and var5.get() == 1):
        predictLR()
        predictID3()
        predknn()
        predbayes()
    elif (var2.get() == 1 and var3.get() == 1 and var4.get() == 1 and var5.get() == 1):
        predictSVM()
        predictID3()
        predknn()
        predbayes()
    elif (var1.get() == 1 and var2.get() == 1 and var3.get() == 1):
        predictLR()
        predictSVM()
        predictID3()
    elif (var1.get() == 1 and var2.get() == 1 and var4.get() == 1):
        predictLR()
        predictSVM()
        predknn()
    elif (var1.get() == 1 and var2.get() == 1 and var5.get() == 1):
        predictLR()
        predictSVM()
        predbayes()
    elif (var1.get() == 1 and var3.get() == 1 and var4.get() == 1):
        predictLR()
        predictID3()
        predknn()
    elif (var1.get() == 1 and var3.get() == 1 and var5.get() == 1):
        predictLR()
        predictID3()
        predbayes()
    elif (var2.get() == 1 and var3.get() == 1 and var4.get() == 1):
        predictSVM()
        predictID3()
        predknn()
    elif (var2.get() == 1 and var3.get() == 1 and var5.get() == 1):
        predictSVM()
        predictID3()
        predbayes()
    elif (var2.get() == 1 and var4.get() == 1 and var5.get() == 1):
        predictSVM()
        predknn()
        predbayes()
    elif (var1.get() == 1 and var4.get() == 1 and var5.get() == 1):
        predictLR()
        predknn()
        predbayes()
    elif (var1.get() == 1 and var2.get() == 1):
        predictLR()
        predictSVM()
    elif (var1.get() == 1 and var3.get() == 1):
        predictLR()
        predictID3()
    elif (var1.get() == 1 and var4.get() == 1):
        predictLR()
        predknn()
    elif (var1.get() == 1 and var5.get() == 1):
        predictLR()
        predbayes()
    elif (var2.get() == 1 and var3.get() == 1):
        predictSVM()
        predictID3()
    elif (var2.get() == 1 and var4.get() == 1):
        predictSVM()
        predknn()
    elif (var2.get() == 1 and var5.get() == 1):
        predictSVM()
        predbayes()
    elif (var3.get() == 1 and var4.get() == 1):
        predictID3()
        predknn()
    elif (var3.get() == 1 and var5.get() == 1):
        predictID3()
        predbayes()
    elif (var1.get() == 1):
        predictLR()
    elif (var2.get() == 1):
        predictSVM()
    elif (var3.get() == 1):
        predictID3()
    elif (var4.get() == 1):
        predknn()
    elif (var5.get() == 1):
        predbayes()


############ GUI #################
from tkinter import *
from tkinter import ttk

view = Tk()
view.geometry('900x900+350+50')
view.maxsize(2000, 2000)
view.minsize(400, 400)
view.resizable(True, True)
view.iconbitmap('./stocks/icon.ico')
view.title('Service Cancellation Predictor')


title = Label(text='Service Cancellation Predictor', font=('Akshar', 30))
title.pack();
##photo =PhotoImage(file='./stocks/icon.png')
##pos=Label(view,image=photo,height=200,width=200)
##pos.place(x=10,y=10)
la1 = Label(text='Mehtodology',   font=('Akshar', 20))
la1.place(x=10, y=100)

var1 = IntVar()
check1 = Checkbutton(view, text="Logistic Regression",  font=('Akshar', 16), variable=var1)
check1.place(x=5, y=150)
var2 = IntVar()
check2 = Checkbutton(view, text='SVM',  font=('Akshar', 16), variable=var2)
check2.place(x=300, y=150)
var3 = IntVar()
check3 = Checkbutton(view, text='ID3',  font=('Akshar', 16), variable=var3)
check3.place(x=500, y=150)
var4 = IntVar()
check4 = Checkbutton(view, text='KNN',  font=('Akshar', 16), variable=var4)
check4.place(x=700, y=150)
var5 = IntVar()
check5 = Checkbutton(view, text='Bayes',  font=('Akshar', 16), variable=var5)
check5.place(x=800, y=150)

but1 = Button(view, width=25, background='#b3b3b3',  text='Train', font=('Akshar', 10), padx=20,
              command=train)
but1.place(x=10, y=200)
but2 = Button(view, width=25, background='#b3b3b3',  text='Test', font=('Akshar', 10), padx=20,
              command=test)
but2.place(x=538, y=200)

fr1 = Frame(height=400, width=880)
fr1.place(x=10, y=250)
la2 = Label(fr1, text='Customer Data',   font=('Akshar', 20))
la2.place(x=10, y=10)

la3 = Label(fr1, text='CustomerID',   font=('Akshar', 12), justify='center')
la3.place(x=10, y=50)
ent1 = Entry(fr1)
ent1.place(x=150, y=55)

la4 = Label(fr1, text='Partner',   font=('Akshar', 12), justify='center')
la4.place(x=10, y=80)
ent2 = Entry(fr1)
ent2.place(x=150, y=85)

la5 = Label(fr1, text='Phone Service',   font=('Akshar', 12), justify='center')
la5.place(x=10, y=110)
ent3 = Entry(fr1)
ent3.place(x=150, y=115)

la6 = Label(fr1, text='Online Security',   font=('Akshar', 12), justify='center')
la6.place(x=10, y=140)
ent4 = Entry(fr1)
ent4.place(x=150, y=145)

la7 = Label(fr1, text='Tech Support',   font=('Akshar', 12), justify='center')
la7.place(x=10, y=170)
ent5 = Entry(fr1)
ent5.place(x=150, y=175)

la8 = Label(fr1, text='Contract',   font=('Akshar', 12), justify='center')
la8.place(x=10, y=200)
ent6 = Entry(fr1)
ent6.place(x=150, y=205)

la9 = Label(fr1, text=' Monthly Charges',   font=('Akshar', 12), justify='center')
la9.place(x=10, y=230)
ent7 = Entry(fr1)
ent7.place(x=150, y=235)

la10 = Label(fr1, text='Gender',   font=('Akshar', 12), justify='center')
la10.place(x=300, y=50)
ent8 = Entry(fr1)
ent8.place(x=418, y=55)

la11 = Label(fr1, text='Dependent',   font=('Akshar', 12), justify='center')
la11.place(x=300, y=80)
ent9 = Entry(fr1)
ent9.place(x=418, y=85)

la12 = Label(fr1, text='Multiple Lines',   font=('Akshar', 12), justify='center')
la12.place(x=300, y=110)
ent10 = Entry(fr1)
ent10.place(x=418, y=115)

la13 = Label(fr1, text='Online Backup',   font=('Akshar', 12), justify='center')
la13.place(x=300, y=140)
ent11 = Entry(fr1)
ent11.place(x=418, y=145)

la14 = Label(fr1, text='Streaming TV',   font=('Akshar', 12), justify='center')
la14.place(x=300, y=170)
ent12 = Entry(fr1)
ent12.place(x=418, y=175)

la15 = Label(fr1, text='Paperless billing',   font=('Akshar', 12), justify='center')
la15.place(x=300, y=200)
ent13 = Entry(fr1)
ent13.place(x=418, y=205)

la16 = Label(fr1, text='Total Charges',   font=('Akshar', 12), justify='center')
la16.place(x=300, y=230)
ent14 = Entry(fr1)
ent14.place(x=415, y=240)

la17 = Label(fr1, text='Senior Citizen',   font=('Akshar', 12), justify='center')
la17.place(x=600, y=50)
ent15 = Entry(fr1)
ent15.place(x=733, y=55)

la18 = Label(fr1, text='Tenure',   font=('Akshar', 12), justify='center')
la18.place(x=600, y=80)
ent16 = Entry(fr1)
ent16.place(x=733, y=85)

la19 = Label(fr1, text='Internet Service',   font=('Akshar', 12), justify='center')
la19.place(x=600, y=110)
ent17 = Entry(fr1)
ent17.place(x=733, y=115)

la20 = Label(fr1, text='Device Protection',   font=('Akshar', 12), justify='center')
la20.place(x=600, y=140)
ent18 = Entry(fr1)
ent18.place(x=733, y=145)

la21 = Label(fr1, text='Streaming Movies',   font=('Akshar', 12), justify='center')
la21.place(x=600, y=170)
ent19 = Entry(fr1)
ent19.place(x=733, y=175)

la22 = Label(fr1, text='Payment Method',   font=('Akshar', 12), justify='center')
la22.place(x=600, y=200)
ent20 = Entry(fr1)
ent20.place(x=733, y=205)

but3 = Button(fr1, width=25, background='#b3b3b3',  text='Predict', font=('Akshar', 10), padx=20,
              command=dataframes)
but3.place(x=350, y=350)

view.mainloop()