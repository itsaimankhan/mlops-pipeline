
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from data_processing import load_and_preprocess

def train():
    X_train, X_test, y_train, y_test = load_and_preprocess()
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)

    joblib.dump(model, "model.joblib")
    print(f"Model trained with accuracy: {accuracy * 100:.2f}%")
    return accuracy

if __name__ == "__main__":
    train()
