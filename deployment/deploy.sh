if [ "$#" -ne 4 ]; then
    echo "Usage : $(basename $0) [REGISTRY] [IMAGE_NAME] [CI_JOB_TOKEN] [APP_DIR]"
    exit 1
fi

REGISTRY=$1
IMAGE_NAME=$2
CI_JOB_TOKEN=$3
APP_DIR=$4

set -e

sudo docker login -u gitlab-ci-token -p $CI_JOB_TOKEN $REGISTRY

sudo docker pull $IMAGE_NAME

cd $APP_DIR

echo "IMAGE_NAME=$IMAGE_NAME" > .env

echo "stop running containers"
sudo docker compose down

echo "Start docker compose..."
sudo docker compose up -d

echo "docker compose ps..."
sudo docker compose ps
