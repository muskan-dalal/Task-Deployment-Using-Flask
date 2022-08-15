import numpy as np
import pandas as pd
import math
from mlxtend.preprocessing import minmax_scaling
from scipy import stats as st
import gunicorn


df = pd.read_csv("prosperLoanData.csv")

#Remove outstanding loans
df = df[df["LoanStatus"] != "Current"]
df["LoanStatus"].value_counts()

#Encode all completed loans as 1, and all delinquent, chargedoff, cancelled and defaulted loans as 0
df["LoanStatus"] = (df["LoanStatus"] == "Completed").astype(int)
df.drop(["ListingKey", "ListingNumber", "LoanKey", "LoanNumber",'LoanFirstDefaultedCycleNumber',"MemberKey","GroupKey"], axis=1, inplace=True)

categorical = df.select_dtypes(include=['bool','object']).columns
numerical=df.select_dtypes('number').columns
df_c = df[categorical].copy()
df_n = df[numerical].copy()

#numerical data handling with mean
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy="mean")
imp.fit(df_n)
df_num_imputed = imp.transform(df_n)

#categorical data with mode
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
imp.fit(df_c)
df_cat_imputed = imp.transform(df_c)

#categorical data with mode
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy="most_frequent")
imp.fit(df_c)
df_cat_imputed = imp.transform(df_c)

#concat c and n data
df_c= pd.DataFrame(df_cat_imputed, columns=df_c.columns.tolist())
df_n= pd.DataFrame(df_num_imputed, columns=df_n.columns.tolist())
data=pd.concat([df_n,df_c],axis=1)

from sklearn.preprocessing import LabelEncoder
cat_list = []
num_list = []
for colname, colvalue in df_c.iteritems():
        cat_list.append(colname)
for col in cat_list:
    encoder = LabelEncoder()
    encoder.fit(df_c[col])
    df_c[col] = encoder.transform(df_c[col])

L=df_c.columns.to_list()
df_c = pd.DataFrame(df_c, columns=L)
data=pd.concat([df_c,data],axis=1)

data = data.select_dtypes(exclude=['object'])

X = data.copy()
y = X.pop("LoanStatus")



X=X.loc[:,['DebtToIncomeRatio', 'StatedMonthlyIncome', 'LoanOriginalAmount',
       'MonthlyLoanPayment', 'LP_CustomerPayments',
       'LP_CustomerPrincipalPayments',
       'LP_GrossPrincipalLoss']]

from sklearn.model_selection import train_test_split
# separate dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=int(len(X) * 0.67),random_state=42)

from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier() # Decision Tree classifer object

# Training Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predicting the response for test dataset
y_pred = clf.predict(X_test)

import pickle
filename='model.pkl'
pickle.dump(clf, open(filename, 'wb'))
