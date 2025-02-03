import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
np.random.seed(0)
X1 = np.random.rand(100) * 10 # Independent variable 1
X2 = np.random.rand(100) * 20 # Independent variable 2
Y = 3 * X1 + 2 * X2 + np.random.randn(100) * 5 # Y = 3*X1 + 2*X2 + noise
X = np.column_stack((X1, X2))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,
random_state=42)
model = LinearRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Mean Squared Error:", mse)
print("R-squared:", r2)import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
np.random.seed(0)
X1 = np.random.rand(100) * 10 # Independent variable 1
X2 = np.random.rand(100) * 20 # Independent variable 2
Y = 3 * X1 + 2 * X2 + np.random.randn(100) * 5 # Y = 3*X1 + 2*X2 + noise
X = np.column_stack((X1, X2))
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,
random_state=42)
model = LinearRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Mean Squared Error:", mse)
print("R-squared:", r2)