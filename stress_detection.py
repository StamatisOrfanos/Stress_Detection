# Import libraries
import os, warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from keras import Input
from xgboost import XGBClassifier
import mlflow
import mlflow.sklearn
from mlflow.models.signature import ModelSignature, infer_signature


warnings.filterwarnings('always')
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Import Data Paths
data_path = os.getcwd() + '/data' 


def data_loader(dataset_name: str, dataset: str):
    '''
    Unify the data loading process to simplify training process, changing column names and types
    Args:
        dataset_name (str): Dataset name provided at each step
        dataset (str): Dataset path to load using pandas 
    '''
    data = pd.read_csv(dataset)
    data.dropna(inplace=True)
    
    if dataset_name == 'Heart Rate Prediction to Monitor Stress Level':
        data = data.rename(columns={'RMSSD': 'HRV'})
        data['condition'] = data['condition'].map({'no stress': 'low', 'interruption': 'medium', 'time pressure': 'high'})
        data = data.rename(columns={'condition': 'label'})
    elif dataset_name == 'SWELL Dataset':
        data['RMSSD'] = data['RMSSD'].astype(int)
        data['HR'] = data['HR'].astype(int)
        data = data.rename(columns={'RMSSD': 'HRV'})
        data['condition'] = data['condition'].map({'no stress': 'low', 'interruption': 'medium', 'time pressure': 'high'})
        data = data.rename(columns={'condition': 'label'})
    
    return data




if __name__ == '__main__':
    
    datasets = {
        'Nurse Stress Prediction Wearable Sensors'      : f'{data_path}/Healthcare/hrv.csv', 
        'Heart Rate Prediction to Monitor Stress Level' : f'{data_path}/Heart_Rate_Prediction/Train_Data/train.csv', 
        'Stress Predict'                                : f'{data_path}/Stress_predict/hrv.csv', 
        'SWELL Dataset'                                 : f'{data_path}/SWELL/train.csv', 
    }


    # Define models
    models = {
        'Logistic Regression':     LogisticRegression(max_iter=100, class_weight='balanced', random_state=26),
        'Random Forest':           RandomForestClassifier(n_estimators=10, class_weight='balanced', random_state=26, min_samples_leaf=1, max_features='sqrt'),
        'SVM (Linear Kernel)':     SVC(kernel='linear', class_weight='balanced', C=1.0, gamma='scale', random_state=26),
        'SVM (RBF Kernel)':        SVC(kernel='rbf', class_weight='balanced', C=1.0, gamma='scale', random_state=26),
        'SVM (Polynomial Kernel)': SVC(kernel='poly', class_weight='balanced', C=1.0, gamma='scale', random_state=26),
        'XGBoost':                 XGBClassifier(eval_metric='logloss', class_weight='balanced'),
        'KNN':                     KNeighborsClassifier(n_neighbors=5),
    }
    
    
    for dataset_name, dataset in datasets.items():
        
        # Create a different experiment for each dataset to save each model and the results under a different run
        mlflow.set_tracking_uri(f'{dataset_name}')
        
        # Load dataset
        data = data_loader(dataset_name, dataset)
        
        if dataset_name == 'Nurse Stress Prediction Wearable Sensors':
            X = data[['HR', 'HRV']]
            y = data['label']

            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
            
            for model_name, model in models.items():
                
                with mlflow.start_run(run_name=model_name):
                    # Train machine learning models and predict
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    # Metrics
                    acc = accuracy_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    
                    accuracy  = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted')
                    recall    = recall_score(y_test, y_pred, average='weighted')
                    f1        = f1_score(y_test, y_pred, average='weighted')

                    # Log parameters and metrics
                    tags = {'data': dataset_name, 'model': model_name}
                    metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
                    mlflow.set_tags(tags)
                    mlflow.log_metrics(metrics)
                    
                    # Enforce signature
                    signature = infer_signature(X_test, y_pred)
                    input_example = {'columns':np.array(X_test.columns), 'data': np.array(X_test.values)}
                    

                    # Log model
                    mlflow.sklearn.log_model(model, model_name) # type: ignore

                    print(f"[{dataset_name}] {model_name} - Accuracy: {acc:.4f} - F1: {f1:.4f}")
                mlflow.end_run()
            
            










