import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

knn_model = joblib.load("knn_model.pkl")
print("Model loaded successfully!\n")

feature_names = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
]

print("Choose an input method:")
print("1. Upload a CSV file")
print("2. Enter data manually")
choice = input("Enter 1 or 2: ")

if choice == "1":
    file_path = input("\nEnter the path of the CSV file: ")
    try:
        data = pd.read_csv(file_path)
        print("\nFile loaded successfully!\n")
        missing_features = [feature for feature in feature_names if feature not in data.columns]
        if missing_features:
            print(f"Error: Missing features in CSV file: {missing_features}")
            exit()
        new_data = data[feature_names]
    except Exception as e:
        print(f"Error loading file: {e}")
        exit()

elif choice == "2":
    print("\nPlease enter the values for the following features:")
    user_input = []
    for feature in feature_names:
        value = float(input(f"{feature}: "))
        user_input.append(value)
    new_data = np.array(user_input).reshape(1, -1)

else:
    print("Invalid choice. Please restart the program and enter 1 or 2.")
    exit()

scaler = StandardScaler()
new_data_scaled = scaler.fit_transform(new_data)

predictions = knn_model.predict(new_data_scaled)

if choice == "1":
    print("\nPrediction Results:")
    for i, prediction in enumerate(predictions):
        diagnosis = "Malignant (Cancerous)" if prediction == 1 else "Benign (Non-Cancerous)"
        print(f"Sample {i+1}: {diagnosis}")
else:
    diagnosis = "Malignant (Cancerous)" if predictions[0] == 1 else "Benign (Non-Cancerous)"
    print("\nPrediction Result:", diagnosis)
