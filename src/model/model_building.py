import numpy as np
import pandas as pd
import os
import pickle
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.metrics import accuracy_score
from src.logger import logging

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

def train_model(X_train: np.ndarray, y_train: np.ndarray) -> BertForSequenceClassification:
    """Train a BERT model for sequence classification."""
    try:
        # Check if CUDA is available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logging.info(f'Using device: {device}')
        
        # Load pretrained BERT model for sequence classification
        model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=len(np.unique(y_train)),
            output_attentions=False,
            output_hidden_states=False,
        )
        model.to(device)
        
        # Convert data to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        
        # Create DataLoader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=16,
            shuffle=True
        )
        
        # Define optimizer
        optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
        
        # Training loop
        epochs = 15
        for epoch in range(epochs):
            logging.info(f'Starting epoch {epoch+1}/{epochs}')
            
            total_loss = 0
            model.train()
            
            for batch in train_dataloader:
                batch_X, batch_y = batch
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                model.zero_grad()
                
                # Forward pass
                # For fine-tuning with pre-extracted BERT embeddings
                # We need to reshape the input to match BERT's expected format
                outputs = model(inputs_embeds=batch_X.view(batch_X.shape[0], 1, batch_X.shape[1]), labels=batch_y)
                
                loss = outputs.loss
                total_loss += loss.item()
                
                # Backward pass
                loss.backward()
                
                # Update weights
                optimizer.step()
            
            avg_loss = total_loss / len(train_dataloader)
            logging.info(f'Epoch {epoch+1} completed with average loss: {avg_loss:.4f}')
        
        logging.info('Model training completed')
        return model
    
    except Exception as e:
        logging.error('Error during model training: %s', e)
        raise

def save_model(model, file_path: str) -> None:
    """Save the trained model to a file."""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save the model
        torch.save(model.state_dict(), file_path)
        
        # Also save model config for later use
        model_info = {
            'model_type': 'bert',
            'pretrained_model_name': 'bert-base-uncased',
        }
        with open(f"{os.path.splitext(file_path)[0]}_info.pkl", 'wb') as f:
            pickle.dump(model_info, f)
            
        logging.info('Model saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the model: %s', e)
        raise

def main():
    try:
        # Load the processed data with BERT embeddings
        train_data = load_data('./data/processed/train_bert.csv')
        
        # Separate features and target
        X_train = train_data.iloc[:, :-1].values
        y_train = train_data.iloc[:, -1].values
        
        # Train the model
        model = train_model(X_train, y_train)
        
        # Save the model
        save_model(model, 'models/bert_model.pt')
        
    except Exception as e:
        logging.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()