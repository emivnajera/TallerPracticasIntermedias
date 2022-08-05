import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures 

df = pd.read_csv("nac.csv")

varx = "Anio"
vary = "Republica"
nb_degree = 2


X = df[varx]
Y = df[vary]
X = np.asanyarray(X)
Y = np.asanyarray(Y)
X = X[:, np.newaxis]
Y = Y[:, np.newaxis]

polynomial_features = PolynomialFeatures(degree=nb_degree)

X_TRANSF = polynomial_features.fit_transform(X)

model = LinearRegression()
model.fit(X_TRANSF, Y)

Y_NEW = model.predict(X_TRANSF)

X_new_min = float(X[0])
X_new_max = float (X[-1])

X_NEW = np.linspace(X_new_min, X_new_max, 50)

X_NEW = X_NEW[:, np.newaxis]

X_NEW_TRANSF = polynomial_features.fit_transform(X_NEW)
Y_NEW = model.predict(X_NEW_TRANSF)

plt.scatter(X_NEW, Y_NEW, color='blue', linewidth = 3)

plt.show()

X_new_min = 0.0
X_new_max = float (2020)

X_NEW = np.linspace(X_new_min, X_new_max, 50)
X_NEW = X_NEW[:, np.newaxis]

X_NEW_TRANSF = polynomial_features.fit_transform(X_NEW)
Y_NEW = model.predict(X_NEW_TRANSF)
print(Y_NEW[-1])

