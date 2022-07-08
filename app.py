from flask import Flask, request, render_template
import pickle
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd

df = pd.read_csv("LOAN-DATASET.csv")
app = Flask(__name__)

# Deserialize
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def index():
    return render_template(
        "index.html")  # due to this function we are able to send our webpage to client(browser) - GET


@app.route('/predict',
           methods=['POST', 'GET'])  # gets inputs data from client(browser) to Flask Server - to give to ml model
def predict():
    df = pd.read_csv("LOAN-DATASET.csv")

    features = [int(x) for x in request.form.values()]
    print(features)
    final = [np.array(features)]
    
    df = df.drop(['ApplicationIncome', 'CoapplicationIncome'], axis=1)
    df['Dependents'].replace('3+', 3, inplace=True)
    df['Loan Status'].replace('N', 0, inplace=True)
    df['Loan Status'].replace('Y', 1, inplace=True)

    df.isnull().sum()

    df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
    df['Married'].fillna(df['Married'].mode()[0], inplace=True)
    df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
    df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
    df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)
    df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0], inplace=True)
    df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)

    df = df.dropna()
    df = df.drop('Loan-ID', axis=1)

    x = df.drop('Loan Status', 1)
    y = df['Loan Status']

    x = pd.get_dummies(x)
    df = pd.get_dummies(df)
    print(y)
    print(x)
    print(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    model = LogisticRegression()
    model.fit(x_train, y_train)
    output = model.predict(x_test)
    print(output)
    print(f"output is {output[0]}")

    if output[0] == 0:
        return render_template('index.html', pred='Not Eligible for Loan')
    elif output[0] == 1 and features != []:
        return render_template('index.html', pred='Eligible for Loan')
    else:
        return render_template('index.html', pred='PLEASE ENTER ALL DETAILS TO PREDICT THE LOAN')


if __name__ == '__main__':
    app.run(debug=True)
