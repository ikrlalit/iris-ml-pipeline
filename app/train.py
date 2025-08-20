# app/model.py
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_and_save_model():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, random_state=42)

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)

    joblib.dump(clf, "iris_model.pkl")
    print("Model saved as iris_model.pkl")

if __name__ == "__main__":
    train_and_save_model()

