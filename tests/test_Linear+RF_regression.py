import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score

def train_and_evaluate_model(X_train, X_test, y_train, y_test, model):
    scaler = RobustScaler()  # or MinMaxScaler

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mse, r2

# Load the data
data = pd.read_csv('C:/Users/Sophie/copro/example/OUT/_REF/DF_out_exgeometry.csv')

X = data.drop(['net_migration', 'poly_ID'], axis=1)
y = data['net_migration']

# Define model types
model_types = [LinearRegression(), RandomForestRegressor(n_estimators=100)]

# Initialize variables to track average R2
total_r2_linear = 0.0
total_r2_rf = 0.0

# Loop for running with different train-test splits and model types
NUMBER_OF_RUNS = 5  # Adjust as needed
for _ in range(NUMBER_OF_RUNS):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # Remove random_state for different splits
    
    for model in model_types:
        mse, r2 = train_and_evaluate_model(X_train, X_test, y_train, y_test, model)
        print(f"Model: {model.__class__.__name__}")
        print(f"Mean Squared Error: {mse}")
        print(f"R-squared Score: {r2}")
        print("=" * 40)  # Separate output for each model type
        
        if isinstance(model, LinearRegression):
            total_r2_linear += r2
        else:
            total_r2_rf += r2

# Calculate average R2 scores
average_r2_linear = total_r2_linear / NUMBER_OF_RUNS
average_r2_rf = total_r2_rf / NUMBER_OF_RUNS

print(f"Average R-squared (Linear Regression): {average_r2_linear}")
print(f"Average R-squared (Random Forest): {average_r2_rf}")






