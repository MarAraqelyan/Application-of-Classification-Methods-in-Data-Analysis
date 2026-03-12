import pandas as pd

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

iris = load_iris()

X = iris.data

y = iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training data count:", len(X_train))
print("Test data count:", len(X_test))


log_model = LogisticRegression()

log_model.fit(X_train, y_train)

y_pred_log = log_model.predict(X_test)

# print("Real labels:")
# print(y_test)

# print("Predicted class names:")
# predicted_names = [str(iris.target_names[p]) for p in y_pred_log]
# print(predicted_names)

# print("Accuracy:", accuracy_score(y_test, y_pred_log))
# print("Confusion matrix:")
# print(confusion_matrix(y_test, y_pred_log))
# print("Classification report:")
# print(classification_report(y_test, y_pred_log, target_names=iris.target_names))



from sklearn.neighbors import KNeighborsClassifier


knn_model = KNeighborsClassifier(n_neighbors=5)

knn_model.fit(X_train, y_train)

y_pred_knn = knn_model.predict(X_test)


print("Real labels:")
print(y_test)

print("Predicted class names:")
predicted_names = [str(iris.target_names[p]) for p in y_pred_knn]
print(predicted_names)

print("Accuracy:", accuracy_score(y_test, y_pred_knn))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred_knn))
print("Classification report:")
print(classification_report(y_test, y_pred_knn, target_names=iris.target_names))