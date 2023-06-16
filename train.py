import mlflow
from mlflow.models.signature import infer_signature

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


mlflow.set_tracking_uri("metadata/")
mlflow.set_experiment("diabetes-experiments-demo")
with mlflow.start_run() as run:
    # Load the diabetes dataset.
    db = load_diabetes()
    X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

    ### TODO: Experiments (change/add) below hyperparameters
    # Set hyperparameters
    params = {'n_estimators':1000, 'max_depth':10, 'max_features':3}

    # Create and train models.
    print("[INFO] Training...")
    rf = RandomForestRegressor(**params)
    rf.fit(X_train, y_train)

    # Use the model to make predictions on the test dataset.
    test_preds = rf.predict(X_test)
    train_preds = rf.predict(X_train)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, train_preds)
    test_mse = mean_squared_error(y_test, test_preds)
    print("[INFO] Train MSE:", train_mse)
    print("[INFO] Test MSE:", test_mse)

    # Add logs to MLflow
    signature = infer_signature(X_test, test_preds)
    mlflow.sklearn.log_model(rf, "model", signature=signature)
    for param in params:
        mlflow.log_param(param, params[param])
    mlflow.log_metric("train_mse", train_mse)
    mlflow.log_metric("test_mse", test_mse)

    print("[INFO]Run ID: {}".format(run.info.run_id))