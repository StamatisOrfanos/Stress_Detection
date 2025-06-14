# Import libraries
import os, warnings, joblib
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, f1_score, precision_score, recall_score, f1_score, classification_report
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import mlflow, mlflow.sklearn
from mlflow.models.signature import ModelSignature, infer_signature
from mlflow.pyfunc import PythonModel # type: ignore
warnings.filterwarnings('always')
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Import Data Paths
data_path = os.getcwd() + '/data' 


class SklearnWrapper(PythonModel):
    def load_context(self, context):
        self.model = joblib.load(context.artifacts["model"])

    def predict(self, context, model_input):
        return self.model.predict(model_input)


def data_loader(dataset_name: str, dataset: str):
    '''
    Unify the data loading process to simplify training process, changing column names and types
    Args:
        dataset_name (str): Dataset name provided at each step
        dataset (str): Dataset path to load using pandas 
    '''
    data = pd.read_csv(dataset)
    data.dropna(inplace=True)
    
    # Standardize HRV column
    if 'RMSSD' in data.columns:
        data = data.rename(columns={'RMSSD': 'HRV'})

    # Standardize label/condition column
    if 'condition' in data.columns:
        data['condition'] = data['condition'].map({'no stress': 'low', 'interruption': 'medium', 'time pressure': 'high'})
        data = data.rename(columns={'condition': 'label'})
    elif 'stress' in data.columns:
        data = data.rename(columns={'stress': 'label'})
    elif 'Label' in data.columns:
        data = data.rename(columns={'Label': 'label'})
    
    # For SWELL Dataset, ensure correct types
    if dataset_name == 'SWELL Dataset':
        if 'HRV' in data.columns:
            data['HRV'] = data['HRV'].astype(int)
        if 'HR' in data.columns:
            data['HR'] = data['HR'].astype(int)
    
    # Ensure 'label' column exists for downstream code
    if 'label' not in data.columns:
        raise ValueError(f"'label' column not found after processing {dataset_name}. Columns are: {data.columns.tolist()}")
    
    return data




if __name__ == '__main__':
    
    datasets = {
        # 'Nurse Stress Prediction Wearable Sensors'      : f'{data_path}/Healthcare/hrv.csv', 
        'Heart Rate Prediction to Monitor Stress Level' : f'{data_path}/Heart_Rate_Prediction/Train_Data/train.csv', 
        'Stress Predict'                                : f'{data_path}/Stress_predict/hrv.csv', 
        'SWELL Dataset'                                 : f'{data_path}/SWELL/train.csv', 
    }

    # Define models
    models = {
        'Logistic Regression':     LogisticRegression(max_iter=100, random_state=26),
        'Random Forest':           RandomForestClassifier(n_estimators=10, random_state=26, min_samples_leaf=1, max_features='sqrt'),
        'SVM (Linear Kernel)':     SVC(kernel='linear', C=1.0, gamma='scale', random_state=26),
        'SVM (RBF Kernel)':        SVC(kernel='rbf', C=1.0, gamma='scale', random_state=26),
        'SVM (Polynomial Kernel)': SVC(kernel='poly', C=1.0, gamma='scale', random_state=26),
        'XGBoost':                 XGBClassifier(eval_metric='logloss'),
        'KNN':                     KNeighborsClassifier(n_neighbors=5),
    }
    
    # Store the best model per dataset     
    best_models = []
    
    for dataset_name, dataset in datasets.items():
        
        # Create a different experiment for each dataset to save each model and the results under a different run
        mlflow.set_tracking_uri(f'{dataset_name}')
        
        # Load dataset
        data = data_loader(dataset_name, dataset)
        X = data[['HR', 'HRV']]
        y = data['label']

        X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
            
        for model_name, model in models.items():

            print(f'Training on dataset: {dataset_name} with model: {model_name}')
            experiment = mlflow.set_experiment(experiment_name=model_name)
            print("Experiment_id: {}".format(experiment.experiment_id))
            print("Artifact Location: {}".format(experiment.artifact_location))
            mlflow.start_run()
            
            # Train machine learning models and predict
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Metrics (set zero_division=0 to suppress UndefinedMetricWarning)
            acc = accuracy_score(y_test, y_pred)
            f1  = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            accuracy  = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall    = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1        = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Log parameters and metrics
            tags    = {'data': dataset_name, 'model': model_name}
            metrics = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1}
            mlflow.set_tags(tags)
            mlflow.log_metrics(metrics)
                
            # Get classification report and log it under artifacts folder
            report    = classification_report(y_test, y_pred, target_names=np.unique(y_test), output_dict=True, zero_division=0)
            report_df = pd.DataFrame(report).transpose()
            report_filename = f'{experiment.artifact_location}/classification_report_{model_name}.csv'
            report_df.to_csv(report_filename)
            mlflow.log_artifact(report_filename)
                
            # Enforce signature
            if dataset_name == 'Heart Rate Prediction to Monitor Stress Level':
                X_test = X_test.astype({col: 'float64' for col in X_test.select_dtypes(include='int').columns})
            signature = infer_signature(X_test, y_pred)
            input_example = {'columns':np.array(X_test.columns), 'data': np.array(X_test.values)}
                
            # Save and log model in pickle format
            pkl_path = f'{experiment.artifact_location}/model.pkl'
            joblib.dump(model, pkl_path)
            mlflow.log_artifact(pkl_path)
            mlflow.sklearn.log_model(model, model_name, signature=signature, input_example=X_test.iloc[:5]) # type: ignore

            # Optional: log Docker-ready model using pyfunc
            mlflow.pyfunc.log_model(
            artifact_path='docker_pyfunc',
            python_model=SklearnWrapper(),
            artifacts={'model': pkl_path},
            conda_env=mlflow.sklearn.get_default_conda_env() # type: ignore
            )

            print(f"[{dataset_name}] {model_name} - Accuracy: {acc:.4f} - F1: {f1:.4f}")
            best_models.append({
            'dataset'  : dataset_name,
            'model'    : model_name,
            'f1_score' : f1,
            'accuracy' : accuracy,
            'precision': precision,
            'recall'   : recall
            })
            
            mlflow.end_run()
            break
    
    # Convert to DataFrame and find best per dataset
    results_df = pd.DataFrame(best_models)
    best_per_dataset = results_df.sort_values('f1_score', ascending=False).groupby('dataset').first().reset_index()

    # Save to CSV
    results_csv_path = 'best_models_summary.csv'
    best_per_dataset.to_csv(results_csv_path, index=False)
    print(f"Saved best model summary to {results_csv_path}")
            
            










