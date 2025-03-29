import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier  # Corrected import

# Load dataset
df = pd.read_csv("C:\\Users\\SIDDHARTH U\\Downloads\\data.csv")

# Display basic info
print(df.head())  # Added parentheses
df.info()

# Drop unnecessary columns
df.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)
print(df.shape)

# Convert diagnosis column to numeric values
def diagnosis_value(diagnosis):
    return 1 if diagnosis == "M" else 0

df['diagnosis'] = df['diagnosis'].apply(diagnosis_value)

# Data visualization
sns.lmplot(x='radius_mean', y='texture_mean', hue='diagnosis', data=df)
sns.lmplot(x='smoothness_mean', y='compactness_mean', hue='diagnosis', data=df)
plt.show()

# Feature selection and splitting
X = np.array(df.iloc[:, 1:])  # Exclude diagnosis column
y = np.array(df['diagnosis'])  # Target labels (0 = benign, 1 = malignant)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=10)

# Using sklearn KNN
knn = KNeighborsClassifier(n_neighbors=9)  # Fixed class name and parameter
knn.fit(X_train, y_train)  # Train the model

# Evaluate model performance
accuracy = knn.score(X_test, y_test)  # Get model accuracy
print(f"Model Accuracy: {accuracy:.2f}")
