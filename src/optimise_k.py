# Training Data Preparation
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt  
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

# Load dataset
df = pd.read_csv("C:\\Users\\SIDDHARTH U\\Downloads\\data.csv")

# Display basic information
df.info()

# Drop unnecessary columns
df.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)

# Convert diagnosis column to numeric values
def diagnosis_value(diagnosis):
    return 1 if diagnosis == "M" else 0

df['diagnosis'] = df['diagnosis'].apply(diagnosis_value)    

# Save preprocessed data
df.to_csv("processed_data.csv", index=False)

# Feature Selection and Splitting
df = pd.read_csv("processed_data.csv")
X = np.array(df.iloc[:, 1:])  # Use all columns except the first (diagnosis)
y = np.array(df['diagnosis'])  # 0 = benign, 1 = malignant

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=10)

# Save training and testing sets
np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

# K-Value Optimization
neighbors = list(range(1, 52, 2))
cv_scores = []

for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

# Compute MSE
MSE = [1 - x for x in cv_scores]

# Find optimal k
optimal_k = neighbors[MSE.index(min(MSE))]
print(f"The optimal number of neighbors (k) is {optimal_k}")

# Plot misclassification error vs. k
plt.figure(figsize=(10, 6))
plt.plot(neighbors, MSE, marker='o', linestyle='dashed', color='b')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Misclassification Error')
plt.title('Optimal k using Cross-Validation')
plt.show()


optimal_k
