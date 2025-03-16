# feature engineering
import numpy as np
import pandas as pd
import os
import yaml
from src.logger import logging
import pickle
from transformers import BertTokenizer, BertModel
import torch
from tqdm import tqdm

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logging.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logging.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logging.error('YAML error: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error: %s', e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)
        logging.info('Data loaded and NaNs filled from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logging.error('Failed to parse the CSV file: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the data: %s', e)
        raise

def get_bert_embeddings(texts, tokenizer, model, max_length=128):
    """
    Extract BERT embeddings for a list of texts.
    
    Args:
        texts: List of text strings
        tokenizer: BERT tokenizer
        model: BERT model
        max_length: Maximum sequence length for BERT
        
    Returns:
        numpy array of BERT embeddings
    """
    all_embeddings = []
    
    # Ensure all inputs are strings
    texts = [str(text) for text in texts]
    
    # Process in batches to avoid memory issues
    batch_size = 16
    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting BERT embeddings"):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize the texts
        encoded_input = tokenizer(
            batch_texts,
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
        
        # Extract features
        with torch.no_grad():
            outputs = model.to(device)(**encoded_input)
            
        # Get the [CLS] token embedding (sentence representation)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(embeddings)
    
    return np.vstack(all_embeddings)

def apply_bow(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> tuple:
    """Apply BERT embeddings to the data instead of BOW."""
    try:
        logging.info("Applying BERT embeddings...")
        
        # Load pre-trained BERT model and tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')
        
        # Convert to list of strings explicitly
        X_train = train_data['review'].astype(str).tolist()
        y_train = train_data['sentiment'].values
        X_test = test_data['review'].astype(str).tolist()
        y_test = test_data['sentiment'].values
        
        # Extract BERT features
        X_train_bert = get_bert_embeddings(X_train, tokenizer, model)
        X_test_bert = get_bert_embeddings(X_test, tokenizer, model)
        
        # Create dataframes with BERT embeddings
        train_df = pd.DataFrame(X_train_bert)
        train_df['label'] = y_train
        
        test_df = pd.DataFrame(X_test_bert)
        test_df['label'] = y_test
        
        # Save tokenizer and model references
        os.makedirs('models', exist_ok=True)
        with open('models/bert_tokenizer_model_info.pkl', 'wb') as f:
            pickle.dump({
                'tokenizer_name': 'bert-base-uncased',
                'model_name': 'bert-base-uncased'
            }, f)
        
        logging.info('BERT embeddings applied and data transformed')
        
        return train_df, test_df
    except Exception as e:
        logging.error('Error during BERT embedding transformation: %s', e)
        raise

def save_data(df: pd.DataFrame, file_path: str) -> None:
    """Save the dataframe to a CSV file."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
        logging.info('Data saved to %s', file_path)
    except Exception as e:
        logging.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():
    try:
        params = load_params('params.yaml')
        max_features = params['feature_engineering']['max_features']
        #max_features = 768  # BERT base has 768 dimensions
        
        train_data = load_data('./data/interim/train_processed.csv')
        test_data = load_data('./data/interim/test_processed.csv')
        
        train_df, test_df = apply_bow(train_data, test_data, max_features)
        
        save_data(train_df, os.path.join("./data", "processed", "train_bert.csv"))
        save_data(test_df, os.path.join("./data", "processed", "test_bert.csv"))
    except Exception as e:
        logging.error('Failed to complete the feature engineering process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()