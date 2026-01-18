import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("Advertising_1.csv")
df.head()

# 1️⃣ RAW DATA SCATTER PLOT
plt.scatter(df['TV'], df['sales'])
plt.xlabel("TV Advertising Budget")
plt.ylabel("Sales")
plt.title("Raw Data")
plt.show()


x = df.iloc[:,0:1]
y = df.iloc[:,-1]


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2, random_state = 2
)


from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)


y_pred = lr.predict(X_test)
y_pred

lr.predict(X_test.iloc[1].values.reshape(1,1))


# 2️⃣ SCATTER PLOT + REGRESSION LINE (FULL DATA)
plt.scatter(df["TV"], df["sales"])
plt.plot(x, lr.predict(x), color='Red')
plt.xlabel("TV Advertising Budget")
plt.ylabel("Sales")
plt.title("Regression Line on Full Data")
plt.show()


# 3️⃣ TEST SPLIT SCATTER + REGRESSION LINE
plt.scatter(X_test, y_test)
plt.plot(X_test, lr.predict(X_test), color='Red')
plt.xlabel("TV Advertising Budget")
plt.ylabel("Sales")
plt.title("Test Data with Regression Line")
plt.show()


# 4️⃣ TRAINING SPLIT SCATTER + REGRESSION LINE
plt.scatter(X_train, y_train)
plt.plot(X_train, lr.predict(X_train), color='Red')
plt.xlabel("TV Advertising Budget")
plt.ylabel("Sales")
plt.title("Training Data with Regression Line")
plt.show()


m = lr.coef_
b = lr.intercept_

# y = mx + b
y = m*150 + b
print(y)
