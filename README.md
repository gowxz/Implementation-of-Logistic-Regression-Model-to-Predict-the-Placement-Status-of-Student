# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
```
1.Import the standard libraries.
2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively.
3.Import LabelEncoder and encode the dataset.
4.Import LogisticRegression from sklearn and apply the model on the dataset.
5.Predict the values of array.
6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
7.Apply new unknown values
```
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: RAKSHITHA DEVI J
RegisterNumber: 212221230082 
*/
import pandas as pd
data = pd.read_csv("Placement_Data.csv")
data.head()
data1 = data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1
x = data1.iloc[:,:-1]
x
y = data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:
head 1:
![image](https://user-images.githubusercontent.com/94165326/162564439-fd6cccd6-f686-4b03-9d5a-b77b3607d0d8.png)
head 2:
![image](https://user-images.githubusercontent.com/94165326/162564472-81a90133-0b37-47bf-a73c-69e7a9b3999a.png)
encoder:
![image](https://user-images.githubusercontent.com/94165326/162564507-406e9354-e7a6-4307-a717-db5faddc978d.png)
accuracy:

![image](https://user-images.githubusercontent.com/94165326/162564527-f6270eb9-ad64-4060-bb5e-b034e0850353.png)

confusion:

![image](https://user-images.githubusercontent.com/94165326/162564542-354c7a5b-b487-43a9-a0df-8ddf396c6cac.png)

classification:

![image](https://user-images.githubusercontent.com/94165326/162564562-5532f058-aeac-4e96-825b-e6ea93b8fe03.png)

predict:

![image](https://user-images.githubusercontent.com/94165326/162564581-1a839c9c-f0f8-407a-b07e-306f3ca640d2.png)

lr predict:
![image](https://user-images.githubusercontent.com/94165326/162564592-c02bb4ae-fcf0-4bdb-a832-74e32cc99975.png)








## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
