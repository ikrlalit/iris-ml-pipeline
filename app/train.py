import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import joblib

# Load and prepare the Iris dataset
def load_and_prepare_data():
    data = pd.read_csv('data/iris.csv')
    train, test = train_test_split(data, test_size=0.4, stratify=data['species'], random_state=42)
    X_train = train[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y_train = train.species
    X_test = test[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
    y_test = test.species
    return X_train, y_train, X_test, y_test

# Train the Decision Tree model
def train_model(X_train, y_train):
    mod_dt = DecisionTreeClassifier(max_depth=3, random_state=1)
    mod_dt.fit(X_train, y_train)
    return mod_dt

# Evaluate the model
def evaluate_model(model, X_test, y_test):
    prediction = model.predict(X_test)
    accuracy = metrics.accuracy_score(prediction, y_test)
    print('The accuracy of the Decision Tree is', "{:.3f}".format(accuracy))
    return accuracy

# Create bucket if it doesn't exist and upload model artifacts to Cloud Storage
def save_and_upload_model(model):
    # Save the model
    joblib.dump(model, "artifacts/model.joblib")


def main():
    # Load and prepare data
    X_train, y_train, X_test, y_test = load_and_prepare_data()
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    evaluate_model(model, X_test, y_test)
    
    # Save and upload model artifacts
    save_and_upload_model(model)

if __name__ == "__main__":
    main()


