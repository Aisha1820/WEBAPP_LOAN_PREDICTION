import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

df=pd.read_csv("LOAN-DATASET.csv")
df['Gender'].value_counts(normalize=True).plot.bar(figsize=(20,10),title='Gender')

df=df.drop(['ApplicationIncome', 'CoapplicationIncome'], axis=1)

df['Dependents'].replace('3+', 3,inplace=True)
df['Loan Status'].replace('N', 0,inplace=True)
df['Loan Status'].replace('Y', 1,inplace=True)

df.isnull().sum()

df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
df['Married'].fillna(df['Married'].mode()[0], inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)

df=df.dropna()
df=df.drop('Loan-ID',axis=1)

x=df.drop('Loan Status',1)
y=df['Loan Status']

x=pd.get_dummies(x)
df=pd.get_dummies(df)
print(x)
print(y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)


model = LogisticRegression()
model.fit(x_train, y_train)

pred_test = model.predict(x_test)
print(pred_test)



pickle.dump(model,open('model.pkl','wb')) #we are Serializing our model by creating model.pkl and writing into it by 'wb'
model=pickle.load(open('model.pkl','rb')) #Deserializing - reading the file - "rb"
print("Success loaded")

#Execute this file only once and create the pkl file.
