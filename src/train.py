import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def train_model(X, y):

    # 🔹 Set MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("salary-prediction")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    with mlflow.start_run():

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)

        mlflow.log_metric("r2_score", r2)
        mlflow.sklearn.log_model(model, "model")

    # 🔥 CRITICAL FIX (absolute path)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = os.path.join(BASE_DIR, "model")

    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "model.pkl")
    joblib.dump(model, model_path)

    print(f"✅ Model saved at: {model_path}")

