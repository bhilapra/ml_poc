import pandas as pd
from math import radians, cos, sin, asin, sqrt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import os

# Load data
df = pd.read_csv("deliverytime.csv")

# Data Cleaning
df = df.drop(columns=["Delivery_person_ID"], errors='ignore')
df = df.dropna(subset=[
    "Delivery_person_Age", "Delivery_person_Ratings",
    "Restaurant_latitude", "Restaurant_longitude",
    "Delivery_location_latitude", "Delivery_location_longitude",
    "Type_of_order", "Type_of_vehicle", "Time_taken(min)"
])
df["Delivery_person_Age"] = pd.to_numeric(df["Delivery_person_Age"], errors="coerce")
df["Delivery_person_Ratings"] = pd.to_numeric(df["Delivery_person_Ratings"], errors="coerce")
df = df.dropna().drop_duplicates()

# Feature engineering - Haversine distance
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * 6371 * asin(sqrt(a))

df["Distance_km"] = df.apply(
    lambda r: haversine(
        r["Restaurant_longitude"], r["Restaurant_latitude"],
        r["Delivery_location_longitude"], r["Delivery_location_latitude"]
    ), axis=1
)

df.drop(columns=[
    "Restaurant_latitude", "Restaurant_longitude",
    "Delivery_location_latitude", "Delivery_location_longitude"
], inplace=True)

# One-hot encode categorical features
df = pd.get_dummies(df, columns=["Type_of_order", "Type_of_vehicle"], drop_first=True)

# Prepare features and target
feature_cols = [
    "Delivery_person_Age", "Delivery_person_Ratings", "Distance_km"
] + list(df.filter(like="Type_of_order_").columns) + list(df.filter(like="Type_of_vehicle_").columns)

X = df[feature_cols]
y = df["Time_taken(min)"]

# Train-test split and scale
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models to try
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

best_mae = float("inf")
best_model_name = None
best_model = None

# Train, evaluate and select best
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, preds)
    print(f"{name} MAE: {mae:.4f}")
    if mae < best_mae:
        best_mae = mae
        best_model_name = name
        best_model = model

print(f"Best Model: {best_model_name} with MAE: {best_mae:.4f}")

# Save best model + scaler
joblib.dump(best_model, "train/models/best_model.pkl")
joblib.dump(scaler, "train/models/scaler.pkl")
print("Saved best model and scaler to train/models/")
