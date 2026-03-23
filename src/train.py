import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def train_model(X,y):

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

    with mlflow.start_run():

      model= LinearRegression()

      model.fit(X_train,y_train)

      y_pred=model.predict(X_test)

      r2=r2_score(y_test,y_pred)

      mlflow.log_metric("r2_score",r2)

      mlflow.sklearn.log_model(model,"model")

    os.makedirs("model",exist_ok=True)
    joblib.dump(model,"model/model.pkl")

    
    print("Model Saved to model/model.pkl successfully")




