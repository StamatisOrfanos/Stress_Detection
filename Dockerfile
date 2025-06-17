FROM python:3.10-slim

RUN pip install mlflow scikit-learn xgboost pandas joblib

# Copy model
COPY ./docker_ready /opt/ml/model

# Serve
CMD mlflow models serve -m /opt/ml/model --no-conda --host 0.0.0.0 --port 5000

