import mlflow

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes


mlflow.set_tracking_uri("metadata/")
mlflow.set_experiment("diabetes-experiments-demo") 

# Load data
db = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

# Load model from a specific run
model = mlflow.sklearn.load_model("runs:/ea733db36a1e40d5a3199a598bb76f99/model") ### TODO: change the run ID to experiment with other models
predictions = model.predict(X_test)
print(predictions)