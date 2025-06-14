# Use official MLflow base image
FROM python:3.10-slim

# Set environment variables
ENV MLFLOW_HOME /opt/mlflow
ENV MODEL_PATH /opt/model

# Create working directory
WORKDIR ${MLFLOW_HOME}

# Install required packages
RUN pip install --upgrade pip && \
    pip install mlflow scikit-learn xgboost pandas numpy

# Copy the model into the container
COPY ./mlruns /opt/mlruns
# OR use this to copy only a single model
# COPY ./model_artifact_dir /opt/model

# Expose port for serving
EXPOSE 5000

# Serve model using MLflow (adjust path)
# For full mlruns path:
CMD ["mlflow", "models", "serve", "-m", "/opt/mlruns/0/<run_id>/artifacts/<model_path>", "-h", "0.0.0.0", "-p", "5000"]

# Example if you copy just the model directory:
# CMD ["mlflow", "models", "serve", "-m", "/opt/model", "-h", "0.0.0.0", "-p", "5000"]
