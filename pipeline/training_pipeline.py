import os
import sys
from prefect import flow
import mlflow

mlflow.set_tracking_uri("file:./mlruns")

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocess import preprocess_data
from src.train import train_model


@flow(name="ML Training Pipeline")
def training_pipeline():

    data_path = "data/raw.csv"

    # Preprocessing
    X, y = preprocess_data(
        path=data_path,
        training=True,
        target_col="salary"
    )

    # Training
    train_model(X, y)


if __name__ == "__main__":
    training_pipeline()