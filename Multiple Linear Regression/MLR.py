import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data_set = pd.read_csv("/workspaces/Machine-Leraning/Multiple Linear Regression/50_Startups.csv")

x=data_set.iloc[:,:-1].values
y=data_set.iloc[:,4].values


one_hot_encoder_x=ColumnTransformer([("State", OneHotEncoder(),[3])], remainder="passthrough")
x=one_hot_encoder_x.fit_transform(x)

x=x[:,1:]

x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.3, random_state=0)

reg=LinearRegression()
reg.fit(x_train, y_train)

y_predict=reg.predict(x_test)

print('Training Score', reg.score(x_train, y_train))
print('Testing Score', reg.score(x_test, y_test))