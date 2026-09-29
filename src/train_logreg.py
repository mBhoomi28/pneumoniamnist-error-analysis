import joblib
from sklearn.linear_model import LogisticRegression
from data_utils import load_flat

X_train, y_train = load_flat("train")

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

joblib.dump(model, "results/logreg_baseline.joblib")
print("Trained and saved to results/logreg_baseline.joblib")
print(f"Train accuracy: {model.score(X_train, y_train):.3f}")