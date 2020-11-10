import numpy as np
import dataset
import regression

X, Y = dataset.load_nonlinear_example1()
ex_X = dataset.polynomial2_features(X)
model = regression.LinearRegression()
model.fit(ex_X,Y)

samples = np.arange(0,4,0.1)
x_samples = np.c_[np.ones(len(samples)),samples]
ex_x_samples = dataset.polynomial2_features(x_samples)

import matplotlib.pyplot as plt
plt.scatter(X[:,1],Y)
plt.plot(samples,model.predict(ex_x_samples))
plt.show