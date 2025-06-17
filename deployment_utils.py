import os
import requests
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