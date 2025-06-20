import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt  
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("C:\\Users\\SIDDHARTH U\\Downloads\\data.csv")

df.info()

df.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)

def diagnosis_value(diagnosis):
    return 1 if diagnosis == "M" else 0

df['diagnosis'] = df['diagnosis'].apply(diagnosis_value)    

df.to_csv("processed_data.csv", index=False)

df = pd.read_csv("processed_data.csv")
X = np.array(df.iloc[:, 1:])
y = np.array(df['diagnosis'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=10)

np.save("X_train.npy", X_train)
np.save("X_test.npy", X_test)
np.save("y_train.npy", y_train)
np.save("y_test.npy", y_test)

neighbors = list(range(1, 52, 2))
cv_scores = []

for k in neighbors:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

MSE = [1 - x for x in cv_scores]

optimal_k = neighbors[MSE.index(min(MSE))]
print(f"The optimal number of neighbors (k) is {optimal_k}")

plt.figure(figsize=(10, 6))
plt.plot(neighbors, MSE, marker='o', linestyle='dashed', color='b')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Misclassification Error')
plt.title('Optimal k using Cross-Validation')
plt.show()

optimal_k
