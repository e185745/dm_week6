import dataset
X,Y = dataset.load_linear_example1()
print(X)
print(X[0])
print(Y)

import regression
model = regression.LinearRegression()
print(model.x)

import importlib
importlib.reload(regression)
model = regression.LinearRegression()
model.fit(X,Y)
print(model.theta)

importlib.reload(regression)
model = regression.LinearRegression()
model.fit(X,Y)
print(model.predict(X))

importlib.reload(regression)
model = regression.LinearRegression()
model.fit(X,Y)
print(model.score(X,Y))

print("ここから")
print("1回目")
X,Y = dataset.load_nonlinear_example1()
ex_X = dataset.polynomial2_features(X)
model = regression.LinearRegression()
model.fit(ex_X,Y)
print(model.theta)
print(model.predict(ex_X))
print(model.score(ex_X,Y))


print("2回目")
X,Y = dataset.load_nonlinear_example1()
ex_X = dataset.polynomial3_features(X)

import importlib
importlib.reload(regression)
model = regression.RidgeRegression()
print(model.alpha)
"""
model.fit(ex_X,Y)
print(model.theta)
print(model.predict(ex_X))
print(model.score(ex_X,Y))
"""