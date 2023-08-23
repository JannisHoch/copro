import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('C:/Users/Sophie/copro/tests/DF_out_exgeometry_totalmigration.csv')

X = data.drop('net_migration', axis=1)  # Replace 'target_column_name' with the actual column name
y = data['net_migration']  # Replace 'target_column_name' with the actual column name

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# model = LinearRegression()

# Initialize the Random Forest Regression model
model = RandomForestRegressor(n_estimators=100, random_state=42)  # or linear regression: 

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)

# Calculate the R-squared score
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared Score: {r2}")

# Get the coefficients of each feature
if model == LinearRegression:
    coefficients = model.coef_

# Create a DataFrame to display the coefficients and their importance
    coefficients_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': coefficients})
    coefficients_df = coefficients_df.sort_values(by='Coefficient', ascending=False)

    print("Feature Importance:")
    print(coefficients_df)

else: 
    # Get the feature importances from the Random Forest model
    feature_importances = model.feature_importances_# Create a DataFrame to display the feature importances
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

print("Feature Importance:")
print(importance_df)






