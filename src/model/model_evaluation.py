import numpy as np
import pandas as pd
import pickle
import json
import torch
from transformers import BertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import logging
import mlflow
import mlflow.pytorch
import dagshub
import os
from src.logger import logging


# Below code block is for production use
# -------------------------------------------------------------------------------------
# Set up DagsHub credentials for MLflow tracking
# dagshub_token = os.getenv("CAPSTONE_TEST")
# if not dagshub_token:
#     raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

# os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
# os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# dagshub_url = "https://dagshub.com"
# repo_owner = "akhilcjdata"
# repo_name = "YT-Capston-Project"

# # Set up MLflow tracking URI
# mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# -------------------------------------------------------------------------------------

# -------------------------------------------------------------------------------------


# Set up MLflow tracking with DAGsHub
dagshub.init(repo_owner='akhilcjdata', repo_name='YT-Capston-Project', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/akhilcjdata/YT-Capston-Project.mlflow')
mlflow.set_experiment('Transformer Embeddings Classification Models with distilbert')

def load_model(file_path: str):
    """Load the trained BERT model from a file."""
    try:
        # First load model info to know how to configure the model
        info_path = f"{os.path.splitext(file_path)[0]}_info.pkl"
        with open(info_path, 'rb') as f:
            model_info = pickle.load(f)
        
        # Get number of labels from the training data (assuming binary classification)
        num_labels = 2  
        
        # Initialize model architecture
        model = BertForSequenceClassification.from_pretrained(
            model_info['pretrained_model_name'],
            num_labels=num_labels,
            output_attentions=False,
            output_hidden_states=False
        )
        
        # Load trained parameters
        model.load_state_dict(torch.load(file_path, map_location=torch.device('cpu')))
        model.eval()  # Set model to evaluation mode
        
        logging.info('BERT model loaded from %s', file_path)
        return model, model_info
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model: %s', e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        logging.info('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise

def evaluate_model(model, model_info, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate the BERT model and return the evaluation metrics."""
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        # Convert data to torch tensors
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        
        # Get predictions
        model.eval()
        with torch.no_grad():
            # For pre-extracted BERT embeddings
            outputs = model(inputs_embeds=X_test_tensor.view(X_test_tensor.shape[0], 1, X_test_tensor.shape[1]))
            logits = outputs.logits
            
            # Convert to probabilities
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            y_pred_proba = probabilities[:, 1].cpu().numpy()
            
            # Get predicted class (0 or 1)
            y_pred = np.argmax(logits.cpu().numpy(), axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        logging.info('Model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logging.error('Error during model evaluation: %s', e)
        raise

def save_metrics(metrics: dict, file_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            json.dump(metrics, file, indent=4)
        logging.info('Metrics saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the metrics: %s', e)
        raise

def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logging.debug('Model info saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the model info: %s', e)
        raise

def main():
    mlflow.set_experiment("bert-sentiment-analysis")
    with mlflow.start_run() as run:  # Start an MLflow run
        try:
            # Load the BERT model
            model, model_info = load_model('./models/bert_model.pt')
            
            # Load test data with BERT embeddings
            test_data = load_data('./data/processed/test_bert.csv')
            
            X_test = test_data.iloc[:, :-1].values
            y_test = test_data.iloc[:, -1].values

            # Evaluate the model
            metrics = evaluate_model(model, model_info, X_test, y_test)
            
            # Save metrics to a file
            save_metrics(metrics, 'reports/metrics.json')
            
            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model parameters to MLflow
            mlflow.log_param("model_type", "bert")
            mlflow.log_param("pretrained_model_name", model_info['pretrained_model_name'])
            
            # Log model to MLflow
            mlflow.pytorch.log_model(model, "bert_model")
            
            # Save model info
            save_model_info(run.info.run_id, "bert_model", 'reports/experiment_info.json')
            
            # Log the metrics file to MLflow
            mlflow.log_artifact('reports/metrics.json')

        except Exception as e:
            logging.error('Failed to complete the model evaluation process: %s', e)
            print(f"Error: {e}")

if __name__ == '__main__':
    main()