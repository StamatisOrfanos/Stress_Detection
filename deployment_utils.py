import os
import requests
<<<<<<< HEAD
import zipfile


def init_model_weights(url: str, model_dir: str = 'model_weights', zip_path: str = 'stress_detector_weights.zip'):
    '''
    Download the zip file containing 
    '''
    model_file = os.path.join(model_dir, 'model.pkl')

    if os.path.exists(model_file):
        print('[INFO] Model weights already present.')
        return

    print(f'[INFO] Downloading model from: {url}')
    response = requests.get(url)
    response.raise_for_status()

    with open(zip_path, 'wb') as f:
        f.write(response.content)
    print('[INFO] Download complete.')

    os.makedirs(model_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall('model_temp')

    for root, dirs, files in os.walk('model_temp'):
        for file in files:
            if file == 'model.pkl':
                src = os.path.join(root, file)
                dst = os.path.join(model_dir, 'model.pkl')
                os.rename(src, dst)
                print('[INFO] Model extracted to:', dst)
                break

    os.remove(zip_path)
    os.system('rm -rf model_temp')
=======
from dotenv import load_dotenv
load_dotenv()
    

FIREBASE_URL=os.getenv('URL')
os.getenv('HF_API_KEY')

def download_weights_from_url(url, local_path):
    if not os.path.exists(local_path):
        # Download the weights if they are not already present
        print("Downloading model weights")
        response = requests.get(url)
        with open(local_path, 'wb') as f:
            f.write(response.content)
            
            
if __name__ == '__main__':
    download_weights_from_url(FIREBASE_URL, 'stress_detector_weights.zip')
>>>>>>> ff403a1 (Set up deployment and create app)
