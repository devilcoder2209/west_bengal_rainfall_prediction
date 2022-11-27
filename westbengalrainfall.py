
import pandas as pd

path = "westbengal_rainfall_1901_to_2017.csv"
df = pd.read_csv(path)
df.fillna(value=0, inplace=True)

# Run a Linear Model on the data
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

div_data = np.asarray(df[['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL',
                          'AUG', 'SEP', 'OCT', 'NOV', 'DEC']])

X = None
y = None
for i in range(div_data.shape[1] - 3):
    if X is None:
        X = div_data[:, i:i + 3]
        y = div_data[:, i + 3]
    else:
        X = np.concatenate((X, div_data[:, i:i + 3]), axis=0)
        y = np.concatenate((y, div_data[:, i + 3]), axis=0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ElasticNet Regression

from sklearn import linear_model

reg = linear_model.ElasticNet(alpha=0.5)
np.nan_to_num(X_train)
np.nan_to_num(y_train)
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
mean_absolute_error(y_test, y_pred)
print("Accuracy on training set: {:.3f}".format(reg.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(reg.score(X_test, y_test)))

import matplotlib.pyplot as plt

plt.plot(y_test, color='red', label='Real data')
plt.plot(y_pred, color='blue', label='Predicted data')
plt.title('Prediction')
plt.legend()
plt.show()

########################################## STREAMLIT #############################################

import streamlit as st
import pandas as pd
import numpy as np

df = pd.read_csv(path)

st.dataframe(df)
data2 = pd.DataFrame(np.random.randn(11, 12),
                     columns=['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC'])
st.area_chart(data2)

yt = []
yp = []

st.title("Rainfall prediction of West Bengal")
option = st.sidebar.selectbox(
    'Select Algorithm',
    ('ElasticNet Regression', 'Random Forest Regression', 'Support Vector Regression', 'Ridge Regression',
     'Lasso Regression', 'Artificial Neural Network'))

# st.write('You selected:', option)

level = st.sidebar.slider("Select the level", 1, 100)
# st.text('Selected: {}'.format(level))

if option == 'ElasticNet Regression':
    for i in y_test:
        yt.append(i)

    for j in y_pred:
        yp.append(j)

    st.write("Accuracy on training set:", reg.score(X_train, y_train).round(3))
    st.write("Accuracy on test set:", reg.score(X_test, y_test).round(3))

chart_data = pd.DataFrame({
    'Predicted Data': yp,
    'Real Data': yt
}
)
st.line_chart(chart_data)

####################################################################################################


# Random Forest Regression
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, max_depth=10, n_jobs=1, verbose=2)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
mean_absolute_error(y_test, y_pred)
print("Accuracy on training set: {:.3f}".format(rf.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(rf.score(X_test, y_test)))

import matplotlib.pyplot as plt

plt.plot(y_test, color='red', label='Real data')
plt.plot(y_pred, color='blue', label='Predicted data')
plt.title('Prediction')
plt.legend()
plt.show()

########################################## STREAMLIT #############################################

import streamlit as st
import pandas as pd

yt = []
yp = []

if option == 'Random Forest Regression':
    for i in y_test:
        yt.append(i)

    for j in y_pred:
        yp.append(j)

chart_data = pd.DataFrame({
    'Predicted Data': yp,
    'Real Data': yt
}
)
st.line_chart(chart_data)

####################################################################################################


# Support Vector Machine
from sklearn.svm import SVR

clf = SVR(gamma='auto', C=0.1, epsilon=0.2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
mean_absolute_error(y_test, y_pred)
print("Accuracy on training set: {:.3f}".format(clf.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(clf.score(X_test, y_test)))
import seaborn as sp
import matplotlib.pyplot as plt

plt.plot(y_test, color='red', label='Real data')
plt.plot(y_pred, color='blue', label='Predicted data')
plt.title('Prediction')
plt.legend()
plt.show()
sp.barplot()

########################################## STREAMLIT #############################################

import streamlit as st
import pandas as pd

yt = []
yp = []

if option == 'Support Vector Regression':
    for i in y_test:
        yt.append(i)

    for j in y_pred:
        yp.append(j)

    st.write("Accuracy on training set:", clf.score(X_train, y_train).round(3))
    st.write("Accuracy on test set:", clf.score(X_test, y_test).round(3))
chart_data = pd.DataFrame({
    'Predicted Data': yp,
    'Real Data': yt
}
)
st.line_chart(chart_data)

####################################################################################################


# Ridge Regression
from sklearn.linear_model import Ridge

rdf = Ridge(alpha=0.5, tol=0.001, solver='auto', random_state=42)
rdf.fit(X_train, y_train)
y_pred = rdf.predict(X_test)
mean_absolute_error(y_test, y_pred)
print("Accuracy on training set: {:.3f}".format(rdf.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(rdf.score(X_test, y_test)))

import matplotlib.pyplot as plt

plt.plot(y_test, color='red', label='Real data')
plt.plot(y_pred, color='blue', label='Predicted data')
plt.title('Prediction')
plt.legend()
plt.show()

########################################## STREAMLIT #############################################

if option == 'Ridge Regression':
    for i in y_test:
        yt.append(i)

    for j in y_pred:
        yp.append(j)

    st.write("Accuracy on training set:", reg.score(X_train, y_train).round(3))
    st.write("Accuracy on test set:", reg.score(X_test, y_test).round(3))

chart_data = pd.DataFrame({
    'Predicted Data': yp,
    'Real Data': yt
}
)
st.line_chart(chart_data)

####################################################################################################


# Lasso Regression
from sklearn.linear_model import Lasso

lrf = Lasso(alpha=1.0, fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=1000, tol=0.0001,
            warm_start=False, positive=False, random_state=None)
lrf.fit(X_train, y_train)
y_pred = lrf.predict(X_test)
mean_absolute_error(y_test, y_pred)
print("Accuracy on training set: {:.3f}".format(lrf.score(X_train, y_train)))
print("Accuracy on test set: {:.3f}".format(lrf.score(X_test, y_test)))

import matplotlib.pyplot as plt

plt.plot(y_test, color='red', label='Real data')
plt.plot(y_pred, color='blue', label='Predicted data')
plt.title('Prediction')
plt.legend()
plt.show()

########################################## STREAMLIT #############################################

if option == 'Lasso Regression':
    for i in y_test:
        yt.append(i)

    for j in y_pred:
        yp.append(j)

    st.write("Accuracy on training set:", reg.score(X_train, y_train).round(3))
    st.write("Accuracy on test set:", reg.score(X_test, y_test).round(3))

chart_data = pd.DataFrame({
    'Predicted Data': yp,
    'Real Data': yt
}
)
st.line_chart(chart_data)

####################################################################################################

# Artificial Neural Network

# from tensorflow.keras import Dense
# from tensorflow.keras import Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Initialising the ANN
model = Sequential()

# Adding the input layer and the first hidden layer
model.add(Dense(32, activation='relu', input_dim=3))

# Adding the second hidden layer
model.add(Dense(units=32, activation='relu'))

# Adding the third hidden layer
model.add(Dense(units=32, activation='relu'))
# Adding the output layer

model.add(Dense(units=1))

# model.add(Dense(1))
# Compiling the ANN
model.compile(optimizer='adam', loss='mean_squared_error')

# Fitting the ANN to the Training set
model.fit(X_train, y_train, batch_size=10, epochs=100)

y_pred = model.predict(X_test)

import matplotlib.pyplot as plt

plt.plot(y_test, color='red', label='Real data')
plt.plot(y_pred, color='blue', label='Predicted data')
plt.title('Prediction')
plt.legend()
plt.show()

########################################## STREAMLIT #############################################

if option == 'Artificial Neural Network':
    for i in y_test:
        yt.append(i)

    for j in y_pred:
        yp.append(j)

chart_data = pd.DataFrame({
    'Predicted Data': yp,
    'Real Data': yt
}
)

####################################################################################################
