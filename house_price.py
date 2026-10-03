import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv("train.csv")

data["Bathrooms"] = data["FullBath"] + (data["HalfBath"] * 0.5)

x = data[["GrLivArea", "BedroomAbvGr", "Bathrooms"]]
y = data["SalePrice"]

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

plt.scatter(x_test["GrLivArea"], y_test)
plt.xlabel("Square Footage")
plt.ylabel("Sale Price")
plt.title("Square Footage vs House Price")
plt.show()

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual Price vs Predicted Price")
plt.show()

print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

area = float(input("Enter house square footage: "))
bedrooms = int(input("Enter number of bedrooms: "))
bathrooms = float(input("Enter number of bathrooms: "))

new_house = pd.DataFrame({
    "GrLivArea": [area],
    "BedroomAbvGr": [bedrooms],
    "Bathrooms": [bathrooms]
})

price = model.predict(new_house)

print("Predicted House Price:", price[0])

