import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from utils import preprocess_data

df = pd.read_csv("data/ads_data.csv")
df, encoders = preprocess_data(df)

features = [
    'device_type','location','gender',
    'content_type','ad_topic','ad_target_audience',
    'hour','weekday','click_through_rate','conversion_rate'
]

X = df[features]

y_age = df['age_group']
y_device = df['device_type']
y_location = df['location']
y_cpc = df['cost_per_click']
y_view = df['view_time']

X_train, X_test, y_age_train, y_age_test = train_test_split(
    X, y_age, test_size=0.25, random_state=42
)

y_device_train = y_device.loc[X_train.index]
y_location_train = y_location.loc[X_train.index]
y_cpc_train = y_cpc.loc[X_train.index]
y_view_train = y_view.loc[X_train.index]

age_model = RandomForestClassifier(n_estimators=200, random_state=42)
age_model.fit(X_train, y_age_train)

device_model = RandomForestClassifier(n_estimators=200, random_state=42)
device_model.fit(X_train, y_device_train)

location_model = RandomForestClassifier(n_estimators=200, random_state=42)
location_model.fit(X_train, y_location_train)

cpc_model = RandomForestRegressor(n_estimators=200, random_state=42)
cpc_model.fit(X_train, y_cpc_train)

view_model = RandomForestRegressor(n_estimators=200, random_state=42)
view_model.fit(X_train, y_view_train)

joblib.dump(age_model, "models/age_model.pkl")
joblib.dump(device_model, "models/device_model.pkl")
joblib.dump(location_model, "models/location_model.pkl")
joblib.dump(cpc_model, "models/cpc_model.pkl")
joblib.dump(view_model, "models/viewtime_model.pkl")
joblib.dump(encoders, "models/encoders.pkl")

print("Models trained and saved successfully")
