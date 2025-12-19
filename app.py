import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(
    page_title="Streamlit with ML",
    page_icon="🕹️",
    layout=None)

url = 'https://raw.githubusercontent.com/TheEconomist/big-mac-data/refs/heads/master/output-data/big-mac-adjusted-index.csv'
df = pd.read_csv(url, parse_dates=['date'])

st.header("Big Mac Index Price Prediction")
st.write("raw data")
st.dataframe(df)

selected_features = ['adj_price', 'GDP_bigmac', 'local_price', 'dollar_ex', 'USD', 'EUR', 'GBP', 'JPY', 'CNY']
X = df[selected_features]
y = df['dollar_price']

df_combined = pd.concat([X, y], axis=1)
df_combined.dropna(inplace=True)

X = df_combined[selected_features]
y = df_combined['dollar_price']

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, Y_train)

Y_pred = model.predict(X_test)

st.write("MSE: ", mean_squared_error(Y_test, Y_pred))
st.write("R squared: ", r2_score(Y_test, Y_pred))

# พล็อตกราฟเปรียบเทียบ

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(Y_test, Y_pred, alpha=0.7)
ax.plot([y.min(), y.max()], [y.min(), y.max()], '--', color='red', lw=2, label='Perfect Prediction Line')
ax.set_xlabel('Actual Dollar Price (Y_test)')
ax.set_ylabel('Predicted Dollar Price (Y_pred)')
ax.set_title('Predicted vs. Actual Dollar Price')
ax.legend()
ax.grid(True)

st.pyplot(fig)

df = pd.DataFrame({
    "Actual Dollar Price (Y_test)": Y_test,
    "Predicted Dollar Price (Y_pred)": Y_pred
})

st.header("Predicted vs. Actual Dollar Price")
st.scatter_chart(
    data=df,
    x="Actual Dollar Price (Y_test)",
    y="Predicted Dollar Price (Y_pred)"
)