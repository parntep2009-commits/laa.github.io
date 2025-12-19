import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
st.set_page_config(page_title="My App", page_icon=":shark:", layout="wide")
st.markdown("""<style>.stApp {background-color: green;}""", unsafe_allow_html=True)
st.markdown('<h1><center> Sustainable Waste Data</center></h1>', unsafe_allow_html=True)
st.set_page_config(layout="wide")

url = "sustainable_waste_management_dataset_2024.csv"
df = pd.read_csv(url, parse_dates=['date'])
st.write(df)

selected_features = ['population','collection_capacity_kg','recyclable_kg', 'organic_kg', 'temp_c']
X = df[selected_features]
y = df['waste_kg']

df_combined = pd.concat([X, y], axis=1)
df_combined.dropna(inplace=True)

X = df_combined[selected_features]
y = df_combined['waste_kg']

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

print("MSE: ", mean_squared_error(Y_test, Y_pred))
print("R squared: ", r2_score(Y_test, Y_pred))

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(Y_test, Y_pred, alpha=0.7)
ax.plot([y.min(), y.max()], [y.min(), y.max()], '--', color='red', lw=2, label='Perfect Prediction Line')
ax.set_xlabel('Actual waste')
ax.set_ylabel('Predicted waste')
ax.set_title('Predicted waste vs. Actual waste')
ax.legend()
ax.grid(True)

st.markdown('<h1><center> Prediction Graph</center></h1>', unsafe_allow_html=True)

st.pyplot(fig)

st.markdown('<h1><center> Area Data </center></h1>', unsafe_allow_html=True)
df = pd.DataFrame(
    np.random.randn(20, 3),
    columns=['a', 'b', 'c'])

st.area_chart(df)




