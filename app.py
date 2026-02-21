import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.title("Diabetes Progression Prediction")

diabetes = load_diabetes()

X = diabetes.data
y = diabetes.target

st.write("Dataset Shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write("Mean Squared Error:", round(mse, 2))
st.write("R-squared Score:", round(r2, 2))

fig, ax = plt.subplots(1, 2, figsize=(14, 6))

ax[0].scatter(y_test, y_pred, alpha=0.5)
ax[0].plot([y_test.min(), y_test.max()],
           [y_test.min(), y_test.max()],
           'k--', lw=2)
ax[0].set_xlabel("Actual Values")
ax[0].set_ylabel("Predicted Values")
ax[0].set_title("True vs Predicted Diabetes Progression")

ax[1].scatter(X_test[:, 2], y_pred, alpha=0.7)
ax[1].set_xlabel("BMI Feature")
ax[1].set_ylabel("Predicted Progression")
ax[1].set_title("BMI vs Predicted Diabetes Progression")

st.pyplot(fig)
