import spacy
import pandas as pd
import re
import logging
from src.logger import logging
import os

def preprocess_dataframe(df, col='text'):
    """
    Preprocess text data in a DataFrame using spacy.
    
    Args:
        df (pandas.DataFrame): DataFrame containing text data
        col (str, optional): Name of column containing text to process. Defaults to 'text'.
        
    Returns:
        pandas.DataFrame: DataFrame with preprocessed text
    """
    # Load spacy English model
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    
    # Create a copy of the dataframe to avoid modifying the original
    processed_df = df.copy()
    
    # Function to process a single text
    def process_text(text):
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to string if it's not already
        if not isinstance(text, str):
            text = str(text)
        
        # Lowercase the text
        text = text.lower()
        
        # Process with spacy
        doc = nlp(text)
        
        # Get tokens and filter out punctuation
        tokens = [token.text for token in doc if not token.is_punct]
        
        # Remove any remaining special characters
        cleaned_tokens = [re.sub(r'[^a-zA-Z0-9\s]', '', token) for token in tokens]
        
        # Filter out empty tokens that might result from cleaning
        cleaned_tokens = [token for token in cleaned_tokens if token.strip()]
        
        # Join the tokens back into a string
        processed_text = ' '.join(cleaned_tokens)
        
        return processed_text
    
    # Apply the processing function to the specified column
    processed_df[col] = processed_df[col].apply(process_text)
    
    return processed_df



def main():
    try:
        # Fetch the data from data/raw
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        logging.info('data loaded properly')

        # Transform the data
        train_processed_data = preprocess_dataframe(train_data, 'review')
        test_processed_data = preprocess_dataframe(test_data, 'review')

        # Store the data inside data/processed
        data_path = os.path.join("./data", "interim")
        os.makedirs(data_path, exist_ok=True)
        
        train_processed_data.to_csv(os.path.join(data_path, "train_processed.csv"), index=False)
        test_processed_data.to_csv(os.path.join(data_path, "test_processed.csv"), index=False)
        
        logging.info('Processed data saved to %s', data_path)
    except Exception as e:
        logging.error('Failed to complete the data transformation process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()