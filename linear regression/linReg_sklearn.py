from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


X,y = load_boston(return_X_y=True)

'''X.shape = (506, 13)
   y.shape = (506,)
'''
#split
X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                 random_state=11,
                                                 test_size=0.2)


reg = LinearRegression()
reg.fit(X_train,y_train)

#prediect
y_pre = reg.predict(X_test)


mean_s_e = mean_squared_error(y_true=y_test,y_pred=y_pre)
print("mean square error is {}".format(mean_s_e))
print()

#print coef and nintercept
k = reg.coef_
b = reg.intercept_
print("the coef is {}".format(k))
print("the intercept is {}".format(b))





