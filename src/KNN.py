import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

df = pd.read_csv("C:\\Users\\SIDDHARTH U\\Downloads\\data.csv")

print(df.head())
df.info()

df.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)
print(df.shape)

def diagnosis_value(diagnosis):
    return 1 if diagnosis == "M" else 0

df['diagnosis'] = df['diagnosis'].apply(diagnosis_value)

sns.lmplot(x='radius_mean', y='texture_mean', hue='diagnosis', data=df)
sns.lmplot(x='smoothness_mean', y='compactness_mean', hue='diagnosis', data=df)
plt.show()

X = np.array(df.iloc[:, 1:])
y = np.array(df['diagnosis'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=10)

knn = KNeighborsClassifier(n_neighbors=9)
knn.fit(X_train, y_train)

accuracy = knn.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")
