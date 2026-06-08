# Import Libraries
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# =========================
# STEP 1: LOAD DATASET
# =========================
iris = load_iris()

X = iris.data
y = iris.target

print("===== DATASET INFORMATION =====")
print("Dataset Shape:", X.shape)
print("Feature Names:", iris.feature_names)
print("Target Names:", iris.target_names)
print()

# =========================
# STEP 2: TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =========================
# STEP 3: DECISION TREE MODEL
# =========================
dt_model = DecisionTreeClassifier(random_state=42)

dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_pred)

print("===== DECISION TREE RESULTS =====")
print(f"Accuracy: {dt_accuracy*100:.2f}%")
print()

print("Classification Report:")
print(classification_report(y_test, dt_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, dt_pred))
print()

# =========================
# STEP 4: KNN MODEL
# =========================
knn_model = KNeighborsClassifier(n_neighbors=3)

knn_model.fit(X_train, y_train)

knn_pred = knn_model.predict(X_test)

knn_accuracy = accuracy_score(y_test, knn_pred)

print("===== KNN RESULTS =====")
print(f"Accuracy: {knn_accuracy*100:.2f}%")
print()

# =========================
# STEP 5: COMPARE MODELS
# =========================
print("===== MODEL COMPARISON =====")
print(f"Decision Tree Accuracy : {dt_accuracy*100:.2f}%")
print(f"KNN Accuracy           : {knn_accuracy*100:.2f}%")
print()

if dt_accuracy > knn_accuracy:
    print("Decision Tree performed better.")
elif knn_accuracy > dt_accuracy:
    print("KNN performed better.")
else:
    print("Both models performed equally.")
print()

# =========================
# STEP 6: VISUALIZATION
# =========================
plt.figure(figsize=(8,5))

plt.scatter(
    X[:,0],
    X[:,1],
    c=y
)

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("Iris Dataset Visualization")

plt.show()

# =========================
# STEP 7: SAMPLE PREDICTION
# =========================

sample_flower = [[5.1, 3.5, 1.4, 0.2]]

prediction = dt_model.predict(sample_flower)

flower_name = iris.target_names[prediction[0]]

print("\n===== SAMPLE PREDICTION =====")
print("Input:", sample_flower[0])
print("Predicted Flower Species:", flower_name)