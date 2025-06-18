FROM python:3.10-slim

WORKDIR /app

<<<<<<< HEAD
# Install system tools
=======
# Install required system packages
>>>>>>> ff403a1 (Set up deployment and create app)
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    unzip \
    findutils \
    && rm -rf /var/lib/apt/lists/*

<<<<<<< HEAD
# Copy and install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Pass the FIREBASE_URL to the build step
ARG URL
ENV URL=$URL

# Download weights using deployment_utils.py
RUN python deployment_utils.py && \
    mkdir -p model_weights && \
    unzip stress_detector_weights.zip -d model_temp && \
    find model_temp -type f -name "model.pkl" -exec mv {} model_weights/model.pkl \; && \
    rm -rf model_temp stress_detector_weights.zip

EXPOSE 5002
=======
# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files including the model zip
COPY . .

# Unzip the model zip, find model.pkl no matter where it is, and move it to a flat target path
RUN unzip stress_detector_weights.zip -d model_temp && \
    mkdir -p model_weights && \
    find model_temp -type f -name "model.pkl" -exec mv {} model_weights/model.pkl \; && \
    rm -rf model_temp stress_detector_weights.zip

RUN echo "model_weights content:" && ls -l model_weights

# Expose the FastAPI port
EXPOSE 5005

>>>>>>> ff403a1 (Set up deployment and create app)
CMD ["python", "-u", "main.py"]
