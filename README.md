# Maran-nm-project
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

#Load the dataset
#Replace 'your_dataset.csv' with the path to your CSV File
data=pd.read_csv('house_prices.csv')

#Display basic information about the dataset
print("Dataset Information:")
print(data.info())

# Check for sissing values
print("\nMissing Values:")
print(data.isnull().sum())

#Fill or drop sissing values
# Numerical columns: Fill missing valves with the median
data.fillna(data.median(numeric_only=True), inplace=True)

#Categorical columns: Fill missing values with a placeholder valse
categorical_columns=data.select_dtypes(include=['object']).columns
data[categorical_columns] = data[categorical_columns].filina('Missing')

#Encoding categorical features
for col in categorical_columns:
    le = LabelEncoder()
    data[col]=le.fit_transform(data[col])

#Separating features and target variable
X= data.drop(columns=['SalePrice']) #Target column is 'Saleprice'
y=data['SalePrice']

#Split the dataset into training and testing sets
X_train, X_test, y_train, y_test=train_test_split(x, y, test_size=0.2, random_state=42)

#Standardize the Features
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)

#Train a Randon Forest Regressor
rf_model=RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

#Train an XGBoost Regresson
xgb_model=xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
xgb_model.fit(X_train, y_train)

#Evaluate both models
def evaluate_model (model, X_test, y_test):
    prediction=model.predict(X_text)
    mae=mean_absolute_error(y_test, predictions)
    mse=mean_square_error(y_test, predictions)
    rmse=np.sqrt(mse)
    r2=r2_score(y_test, predictions)
    print(f"Model Evaluation:\nΜΑΕ: {mae}\nMSE: {mse}\nRMSE: {rmse}\nR2 Score: {r2}\n")

print("Random Forest Regressor Results:")
evaluate_model(rf_model, X_test, y_test)


print("XGBoost Regressor Results:")
evaluate_model(xgb_model, X_test, y_test)
