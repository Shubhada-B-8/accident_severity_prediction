import joblib

# Load the weather encoder
le_weather = joblib.load('le_weather.pkl')

print("Weather classes in encoder:")
print(le_weather.classes_)
print(f"\nNumber of classes: {len(le_weather.classes_)}")