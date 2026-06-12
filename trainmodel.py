import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import joblib
import warnings
warnings.filterwarnings('ignore')

# ============================================
# STEP 1: SET CORRECT PATH
# ============================================
print("="*60)
print("STEP 1: Setting up paths")
print("="*60)

current_dir = os.getcwd()
print(f"Current directory: {current_dir}")

# ============================================
# STEP 2: LOAD DATASETS
# ============================================
print("\n" + "="*60)
print("STEP 2: Loading Datasets")
print("="*60)

files = {
    'uk_accidents': 'AccidentsBig.csv',
    'uk_casualties': 'CasualtiesBig.csv',
    'uk_vehicles': 'VehiclesBig.csv'
}

data = {}
for key, fname in files.items():
    path = os.path.join(current_dir, fname)
    if os.path.exists(path):
        try:
            data[key] = pd.read_csv(path)
            print(f"✅ Loaded {key}: {data[key].shape}")
        except Exception as e:
            print(f"❌ Error loading {fname}: {e}")

# ============================================
# STEP 3: CLEAN UK DATASETS
# ============================================
print("\n" + "="*60)
print("STEP 3: Cleaning UK Datasets")
print("="*60)

# Clean UK Accidents
df_uk_acc = data['uk_accidents'].copy()
df_uk_acc = df_uk_acc.dropna(subset=['Accident_Index'])
print(f"UK Accidents after dropping nulls: {df_uk_acc.shape}")

# Extract hour from Time
if 'Time' in df_uk_acc.columns:
    df_uk_acc['hour'] = pd.to_datetime(df_uk_acc['Time'], format='%H:%M', errors='coerce').dt.hour
    df_uk_acc['hour'].fillna(12, inplace=True)
else:
    df_uk_acc['hour'] = 12

# Fill missing values in important columns
important_cols = ['Accident_Severity', 'Number_of_Vehicles', 'Number_of_Casualties', 
                  'Speed_limit', 'Road_Type', 'Light_Conditions', 'Urban_or_Rural_Area']
for col in important_cols:
    if col in df_uk_acc.columns:
        if df_uk_acc[col].dtype == 'object':
            df_uk_acc[col].fillna(df_uk_acc[col].mode()[0], inplace=True)
        else:
            df_uk_acc[col].fillna(df_uk_acc[col].median(), inplace=True)

print(f"✅ UK Accidents cleaned: {df_uk_acc.shape}")

# UK Casualties (aggregate to accident level)
df_uk_cas = data['uk_casualties'].copy()
casualty_agg = df_uk_cas.groupby('Accident_Index').agg({
    'Casualty_Severity': ['count', lambda x: (x == 1).sum()]
}).reset_index()
casualty_agg.columns = ['Accident_Index', 'total_casualties', 'fatal_casualties']
print(f"✅ Casualties aggregated: {casualty_agg.shape}")

# UK Vehicles (aggregate to accident level)
df_uk_veh = data['uk_vehicles'].copy()
df_uk_veh = df_uk_veh.dropna(subset=['Accident_Index'])

# Create vehicle type categories
def categorize_vehicle(vehicle_type):
    if vehicle_type in [1,2,3,4,5]:
        return 'Two_Wheeler'
    elif vehicle_type in [9,10,11,12,13]:
        return 'Car'
    elif vehicle_type in [16,17,18,19]:
        return 'Bus'
    elif vehicle_type in [20,21,22,23,24,25]:
        return 'Heavy_Vehicle'
    else:
        return 'Other'

if 'Vehicle_Type' in df_uk_veh.columns:
    df_uk_veh['vehicle_category'] = df_uk_veh['Vehicle_Type'].apply(categorize_vehicle)

vehicle_agg = df_uk_veh.groupby('Accident_Index').agg({
    'vehicle_category': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown',
    'Age_of_Driver': 'mean'
}).reset_index()
vehicle_agg.columns = ['Accident_Index', 'primary_vehicle_type', 'avg_driver_age']
print(f"✅ Vehicles aggregated: {vehicle_agg.shape}")

# ============================================
# STEP 4: MERGE ALL UK DATA
# ============================================
print("\n" + "="*60)
print("STEP 4: Merging UK Datasets")
print("="*60)

# Start with accidents
df_merged = df_uk_acc.copy()

# Merge with casualties
df_merged = df_merged.merge(casualty_agg, on='Accident_Index', how='left')
df_merged['total_casualties'].fillna(0, inplace=True)
df_merged['fatal_casualties'].fillna(0, inplace=True)

# Merge with vehicles
df_merged = df_merged.merge(vehicle_agg, on='Accident_Index', how='left')
df_merged['primary_vehicle_type'].fillna('Unknown', inplace=True)
df_merged['avg_driver_age'].fillna(40, inplace=True)

print(f"✅ Merged dataset shape: {df_merged.shape}")

# ============================================
# STEP 5: FEATURE ENGINEERING
# ============================================
print("\n" + "="*60)
print("STEP 5: Feature Engineering")
print("="*60)

# Map weather codes to names
def map_weather(code):
    if pd.isna(code):
        return 'Unknown'
    mapping = {1: 'Clear', 2: 'Rain', 3: 'Snow', 4: 'Fog', 5: 'Other'}
    return mapping.get(int(code), 'Other')

if 'Weather_Conditions' in df_merged.columns:
    df_merged['weather'] = df_merged['Weather_Conditions'].apply(map_weather)
else:
    df_merged['weather'] = 'Unknown'

# Map road types
def map_road_type(road_type):
    if pd.isna(road_type):
        return 'Unknown'
    mapping = {1: 'Motorway', 2: 'A_Road', 3: 'B_Road', 4: 'Minor_Road'}
    return mapping.get(int(road_type), 'Other')

if 'Road_Type' in df_merged.columns:
    df_merged['road_type'] = df_merged['Road_Type'].apply(map_road_type)
else:
    df_merged['road_type'] = 'Unknown'

# Map light conditions
def map_light(light_code):
    if pd.isna(light_code):
        return 'Unknown'
    if light_code in [1,2,3]:
        return 'Daylight'
    elif light_code in [4,5,6,7]:
        return 'Dark'
    else:
        return 'Other'

if 'Light_Conditions' in df_merged.columns:
    df_merged['light'] = df_merged['Light_Conditions'].apply(map_light)
else:
    df_merged['light'] = 'Unknown'

# Create derived features
df_merged['casualty_per_vehicle'] = df_merged['Number_of_Casualties'] / (df_merged['Number_of_Vehicles'] + 1)
df_merged['is_night'] = ((df_merged['hour'] < 6) | (df_merged['hour'] > 20)).astype(int)
df_merged['is_rush_hour'] = ((df_merged['hour'].between(7, 9)) | (df_merged['hour'].between(17, 19))).astype(int)

print("✅ Feature engineering complete")

# ============================================
# STEP 6: PREPARE DATA FOR MODELING
# ============================================
print("\n" + "="*60)
print("STEP 6: Preparing Data for Modeling")
print("="*60)

# Select target
df_merged['severity'] = pd.to_numeric(df_merged['Accident_Severity'], errors='coerce')
df_merged = df_merged.dropna(subset=['severity'])
df_merged['severity'] = df_merged['severity'].astype(int)

# Filter to valid severity values (1,2,3)
df_merged = df_merged[df_merged['severity'].isin([1,2,3])]

print(f"Severity distribution:")
print(df_merged['severity'].value_counts().sort_index())

# Encode categorical features
le_weather = LabelEncoder()
le_road = LabelEncoder()
le_light = LabelEncoder()
le_vehicle = LabelEncoder()

df_merged['weather_enc'] = le_weather.fit_transform(df_merged['weather'])
df_merged['road_enc'] = le_road.fit_transform(df_merged['road_type'])
df_merged['light_enc'] = le_light.fit_transform(df_merged['light'])
df_merged['vehicle_enc'] = le_vehicle.fit_transform(df_merged['primary_vehicle_type'])

# Features for modeling
feature_cols = [
    'weather_enc', 'road_enc', 'light_enc', 'vehicle_enc',
    'Speed_limit', 'Number_of_Vehicles', 'Number_of_Casualties',
    'hour', 'avg_driver_age', 'casualty_per_vehicle', 'is_night', 'is_rush_hour'
]

# Make sure all features exist
existing_features = [col for col in feature_cols if col in df_merged.columns]
print(f"Using {len(existing_features)} features: {existing_features}")

X = df_merged[existing_features].copy()
y = df_merged['severity'].copy()

# Remap severity to 0-2 for modeling
severity_mapping = {1:0, 2:1, 3:2}
y = y.map(severity_mapping)

# Fill any missing values
for col in X.columns:
    if X[col].dtype in ['int64', 'float64']:
        X[col] = X[col].fillna(X[col].median())
    else:
        X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 0)

print(f"✅ Features ready: {X.shape}")
print(f"   Target classes: {y.unique()}")

# ============================================
# STEP 7: TRAIN MODEL WITH SMOTE
# ============================================
print("\n" + "="*60)
print("STEP 7: Training Model with SMOTE")
print("="*60)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale numerical features
num_features = ['Speed_limit', 'Number_of_Vehicles', 'Number_of_Casualties', 'hour', 'avg_driver_age', 'casualty_per_vehicle']
num_features = [f for f in num_features if f in X_train.columns]

scaler = StandardScaler()
X_train[num_features] = scaler.fit_transform(X_train[num_features])
X_test[num_features] = scaler.transform(X_test[num_features])

# Apply SMOTE to balance classes
print("Applying SMOTE to balance classes...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
print(f"Before SMOTE: {X_train.shape}")
print(f"After SMOTE: {X_train_balanced.shape}")

# Train Random Forest
print("\nTraining Random Forest...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_balanced, y_train_balanced)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Feature Importance
feature_importance = pd.DataFrame({
    'feature': existing_features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)
print("\nTop 10 Feature Importances:")
print(feature_importance.head(10))

# ============================================
# STEP 8: SAVE MODEL AND FILES
# ============================================
print("\n" + "="*60)
print("STEP 8: Saving Model and Files")
print("="*60)

joblib.dump(model, 'best_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le_weather, 'le_weather.pkl')
joblib.dump(le_road, 'le_road.pkl')
joblib.dump(le_light, 'le_light.pkl')
joblib.dump(le_vehicle, 'le_vehicle.pkl')
joblib.dump(existing_features, 'feature_cols.pkl')
joblib.dump(severity_mapping, 'severity_mapping.pkl')

# Save unified dataset for EDA
df_merged.to_csv('unified_accident_data.csv', index=False)

print("✅ All files saved successfully!")
print(f"\nWeather classes: {list(le_weather.classes_)}")
print(f"Road type classes: {list(le_road.classes_)}")
print(f"Light condition classes: {list(le_light.classes_)}")
print(f"Vehicle type classes: {list(le_vehicle.classes_)}")
print("\n" + "="*60)
print("READY! Now run: streamlit run app.py")
print("="*60)