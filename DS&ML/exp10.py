import pandas as panda
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

cal=fetch_california_housing()
df=panda.DataFrame(data=cal.data,columns=cal.feature_names)
df['Target']=cal.target
x=df.drop('Target',axis=1)
y=df['Target']

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.2,random_state=42)
lr=LinearRegression()
lr.fit(x_train,y_train)
p=lr.predict(x_test)
res=mean_squared_error(y_test,p)
print("MSE : ",res)







