import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, r2_score
import os

def train_and_evaluate_model(X_train, X_test, y_train, y_test):
    scaler = RobustScaler()  # or MinMaxScaler

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    feature_importances = model.feature_importances_

    return mse, r2, feature_importances

data = pd.read_csv(r'C:\Users\Sophie\copro\tests\India\DF_out_exgeometry_northern_India_out.csv')

X = data.drop(['net_migration', 'poly_ID'], axis=1)
y = data['net_migration']

# Initialize variables to track average R2 and feature importance
total_r2_rf = 0.0
total_feature_importance_rf = None

# Loop for running with different train-test splits
NUMBER_OF_RUNS = 100
for _ in range(NUMBER_OF_RUNS):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)  # Remove random_state for different splits
    
    mse, r2, feature_importances = train_and_evaluate_model(X_train, X_test, y_train, y_test)
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared Score: {r2}")
    print("=" * 40)  # Separate output for each run

    total_r2_rf += r2
    if total_feature_importance_rf is None:
        total_feature_importance_rf = feature_importances
    else:
        total_feature_importance_rf += feature_importances

# Calculate average R2 score
average_r2_rf = total_r2_rf / NUMBER_OF_RUNS
data_directory = (r'C:\Users\Sophie\copro\tests\India')

print("\nAverage R-squared Score (Random Forest):", average_r2_rf)

if total_feature_importance_rf is not None:
    # Create a DataFrame for feature importances
    feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': total_feature_importance_rf / NUMBER_OF_RUNS})
    
    # Sort the DataFrame by Importance in descending order
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    
    # Calculate relative importances
    feature_importance_df['Relative Importance'] = feature_importance_df['Importance'] / feature_importance_df['Importance'].iloc[0]
    
    print("\nFeature Importance Table for RandomForestRegressor:")
    print(feature_importance_df)

    # Save the average feature importance to a CSV file
    feature_importance_df.to_csv(os.path.join(data_directory, 'average_feature_importance_rf_northern_India_out.csv'), index=False)

else:
    print("\nFeature Importance Table for RandomForestRegressor: No feature importances available.")

# Save the average R-squared score to a CSV file
r2_df = pd.DataFrame({'Model': ['Random Forest'], 'Average R-squared Score': [average_r2_rf]})
r2_df.to_csv(os.path.join(data_directory, 'average_r2_rf_northern_India_out.csv'), index=False)




