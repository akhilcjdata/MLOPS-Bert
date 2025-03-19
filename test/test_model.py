import unittest
import mlflow
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
from transformers import BertTokenizer

class TestModelLoading(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up DagsHub credentials for MLflow tracking
        dagshub_token = os.getenv("CAPSTONE_TEST")
        if not dagshub_token:
            raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

        os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

        dagshub_url = "https://dagshub.com"
        repo_owner = "akhilcjdata"
        repo_name = "YT-Capstone-Project"

        # Set up MLflow tracking URI
        mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')

        # Load the new model from MLflow model registry
        cls.new_model_name = "my_model"
        cls.new_model_version = cls.get_latest_model_version(cls.new_model_name)
        cls.new_model_uri = f'models:/{cls.new_model_name}/{cls.new_model_version}'
        cls.new_model = mlflow.pyfunc.load_model(cls.new_model_uri)

        # Load BERT tokenizer
        cls.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        # Load holdout test data - assuming we have raw text and labels
        cls.holdout_data = pd.read_csv('data/processed/test_data.csv')

    @staticmethod
    def get_latest_model_version(model_name, stage="Staging"):
        client = mlflow.MlflowClient()
        latest_version = client.get_latest_versions(model_name, stages=[stage])
        return latest_version[0].version if latest_version else None

    def test_model_loaded_properly(self):
        self.assertIsNotNone(self.new_model)

    def test_model_signature(self):
        # Create a dummy input for the model based on BERT's expected input format
        input_text = "hi how are you"
        
        # Tokenize the input text for BERT
        encoded_input = self.tokenizer(
            input_text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='np'
        )
        
        # Convert to DataFrame format expected by MLflow PyFunc model
        input_df = pd.DataFrame({
            'input_ids': [encoded_input['input_ids'][0].tolist()],
            'attention_mask': [encoded_input['attention_mask'][0].tolist()],
            'token_type_ids': [encoded_input['token_type_ids'][0].tolist()]
        })

        # Predict using the new model to verify the input and output shapes
        prediction = self.new_model.predict(input_df)

        # Verify the output shape (assuming binary classification with a single output)
        self.assertEqual(len(prediction), input_df.shape[0])
        self.assertEqual(len(prediction.shape), 1)  # Assuming a single output column for binary classification

    def test_model_performance(self):
        # Extract text and labels from holdout test data
        texts = self.holdout_data['text'].tolist()  # Assuming 'text' column contains the raw text
        y_holdout = self.holdout_data['label'].values  # Assuming 'label' column contains the target values

        # Prepare batch inputs for BERT
        batch_size = 32
        all_predictions = []
        
        # Process in batches to prevent memory issues
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize the batch
            encoded_batch = self.tokenizer(
                batch_texts,
                padding='max_length',
                truncation=True,
                max_length=128,
                return_tensors='np'
            )
            
            # Convert to DataFrame format expected by MLflow PyFunc model
            batch_df = pd.DataFrame({
                'input_ids': encoded_batch['input_ids'].tolist(),
                'attention_mask': encoded_batch['attention_mask'].tolist(),
                'token_type_ids': encoded_batch['token_type_ids'].tolist()
            })
            
            # Get predictions for this batch
            batch_predictions = self.new_model.predict(batch_df)
            all_predictions.extend(batch_predictions)
        
        # Convert predictions to numpy array
        y_pred_new = np.array(all_predictions)

        # Calculate performance metrics for the new model
        accuracy_new = accuracy_score(y_holdout, y_pred_new)
        precision_new = precision_score(y_holdout, y_pred_new)
        recall_new = recall_score(y_holdout, y_pred_new)
        f1_new = f1_score(y_holdout, y_pred_new)

        # Define expected thresholds for the performance metrics
        expected_accuracy = 0.40
        expected_precision = 0.40
        expected_recall = 0.40
        expected_f1 = 0.40

        # Assert that the new model meets the performance thresholds
        self.assertGreaterEqual(accuracy_new, expected_accuracy, f'Accuracy should be at least {expected_accuracy}')
        self.assertGreaterEqual(precision_new, expected_precision, f'Precision should be at least {expected_precision}')
        self.assertGreaterEqual(recall_new, expected_recall, f'Recall should be at least {expected_recall}')
        self.assertGreaterEqual(f1_new, expected_f1, f'F1 score should be at least {expected_f1}')

if __name__ == "__main__":
    unittest.main()