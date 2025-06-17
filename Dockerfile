FROM python:3.10-slim

WORKDIR /app

# Install system tools
RUN apt-get update && apt-get install -y \
    gcc \
    python3-dev \
    unzip \
    findutils \
    && rm -rf /var/lib/apt/lists/*

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

EXPOSE 5005
CMD ["python", "-u", "main.py"]
