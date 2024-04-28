import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Generate some random data for house sizes and prices
np.random.seed(0)
house_sizes = np.random.randint(1000, 5000, 100)  # House sizes in square feet
prices = 100 * house_sizes + np.random.normal(0, 20000, 100)  # Price in dollars

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(house_sizes, prices, test_size=0.2, random_state=42)

# Reshape the data for sklearn
X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test data
predictions = model.predict(X_test)

# Plot the results
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, predictions, color='red')
plt.title('House Price Prediction')
plt.xlabel('House Size (sqft)')
plt.ylabel('Price ($)')
plt.show()

# Print the model's coefficients
print("Coefficients: ", model.coef_)
print("Intercept: ", model.intercept_)
