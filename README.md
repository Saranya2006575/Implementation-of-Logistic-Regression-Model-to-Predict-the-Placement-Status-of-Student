# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load the dataset, drop unnecessary columns, and encode categorical variables. 2.Define the features (X) and target variable (y). 3.Split the data into training and testing sets. 4.Train the logistic regression model, make predictions, and evaluate using accuracy and other
## Program:
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Saranya.J
RegisterNumber:  212224240146
*/
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv(r"C:\Users\admin\Desktop\Placement_Data.csv")
data.head()

datal = data.copy()
datal = datal.drop(["sl_no", "salary"], axis=1)
datal.head()

print("Missing values:\n", datal.isnull().sum())
print("Duplicate rows:", datal.duplicated().sum())

le = LabelEncoder()
datal["gender"] = le.fit_transform(datal["gender"])
datal["ssc_b"] = le.fit_transform(datal["ssc_b"])
datal["hsc_b"] = le.fit_transform(datal["hsc_b"])
datal["hsc_s"] = le.fit_transform(datal["hsc_s"])
datal["degree_t"] = le.fit_transform(datal["degree_t"])
datal["workex"] = le.fit_transform(datal["workex"])
datal["specialisation"] = le.fit_transform(datal["specialisation"])
datal["status"] = le.fit_transform(datal["status"])

x = datal.iloc[:, :-1]
y = datal["status"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

lr = LogisticRegression(solver="liblinear")
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", confusion)

classification_report_output = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_report_output)

## Output:
<img width="1033" height="268" alt="image" src="https://github.com/user-attachments/assets/bb92628e-ffce-43f1-a3f4-7622f80a5ea5" />
<img width="1035" height="312" alt="image" src="https://github.com/user-attachments/assets/9e3ce249-6a1d-41e5-808e-83b45ead975b" />
<img width="1036" height="652" alt="image" src="https://github.com/user-attachments/assets/bebe9c26-0e52-4e50-bf2a-e56946bbbece" />
<img width="1035" height="259" alt="image" src="https://github.com/user-attachments/assets/a7b9158d-e411-4d36-ae12-c5f2f00dae5d" />
<img width="1036" height="144" alt="image" src="https://github.com/user-attachments/assets/b3e1e40b-382a-4eda-9196-bd59125ead83" />
<img width="1032" height="164" alt="image" src="https://github.com/user-attachments/assets/e3b714b6-cab6-4db6-9635-2f0e2e313da2" />
<img width="1035" height="439" alt="image" src="https://github.com/user-attachments/assets/c60da504-38a3-4800-ac28-a2f7ba3c2440" />


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
