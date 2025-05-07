# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages.
2. Analyse the data.
3. Use modelselection and Countvectorizer to preditct the values.
4. Find the accuracy and display the result.
 

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: SANTHIYA R
RegisterNumber:  212223230192
*/

import pandas as pd
data=pd.read_csv("spam.csv", encoding='Windows-1252')
data

data.shape

x=data['v2'].values
y=data['v1'].values
x.shape

y.shape

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=0)
x_train

x_train.shape

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
acc=accuracy_score(y_test,y_pred)
acc

con=confusion_matrix(y_test,y_pred)
print(con)

cl=classification_report(y_test,y_pred)
print(cl)
```


## Output:
### data
![image](https://github.com/user-attachments/assets/c68cfd93-0c3e-45d6-9986-cc9d17fef2e2)

### data.shape()
![image](https://github.com/user-attachments/assets/e7c3ec3b-2692-4fe2-b61c-ebb47b3dddb8)

### x.shape()
![image](https://github.com/user-attachments/assets/b6dc6b43-8ea9-4e83-b01d-f2de57cc9cd3)

### x_train.shape()
![image](https://github.com/user-attachments/assets/3dfdad3f-28b1-49ec-b27e-70cf85742226)

### y_pred
![image](https://github.com/user-attachments/assets/02d3b95e-ef32-464b-a85a-170ec5d68979)


### acc (accuracy)
![image](https://github.com/user-attachments/assets/11f9bcba-e4a2-4e45-a80a-bc2147adb8cf)


### con (confusion matrix)
![image](https://github.com/user-attachments/assets/3150e259-78f6-4e2c-a58c-5858fb20c6f5)

### cl (classification report)
![image](https://github.com/user-attachments/assets/457c20b7-8cda-4097-a3da-848e2b7cd966)








## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
