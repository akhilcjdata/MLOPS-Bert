from flask import Flask, render_template, request
import mlflow
import pickle
import os
import pandas as pd
import numpy as np
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
import time
import re
import spacy
import dagshub
import warnings
import torch
from transformers import AutoTokenizer

warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# Load spaCy model
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

def lemmatization(text):
    """Lemmatize the text using spaCy."""
    if not isinstance(text, str) or text.strip() == '':
        return ''
    
    doc = nlp(text)
    text = [token.lemma_ for token in doc]
    return " ".join(text)

def remove_stop_words(text):
    """Remove stop words from the text using spaCy."""
    if not isinstance(text, str) or text.strip() == '':
        return ''
        
    doc = nlp(text)
    text = [token.text for token in doc if not token.is_stop]
    return " ".join(text)

def removing_numbers(text):
    """Remove numbers from the text."""
    if not isinstance(text, str):
        return ''
        
    text = ''.join([char for char in text if not char.isdigit()])
    return text

def lower_case(text):
    """Convert text to lower case."""
    if not isinstance(text, str):
        return ''
        
    return text.lower()

def removing_punctuations(text):
    """Remove punctuations from the text using spaCy."""
    if not isinstance(text, str) or text.strip() == '':
        return ''
        
    doc = nlp(text)
    text = [token.text for token in doc if not token.is_punct]
    text = ' '.join(text)
    text = re.sub('\s+', ' ', text).strip()
    return text

def removing_urls(text):
    """Remove URLs from the text."""
    if not isinstance(text, str):
        return ''
        
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)

def normalize_text(text):
    """Process text using spaCy pipeline."""
    if not isinstance(text, str) or text.strip() == '':
        return ''
    
    # Apply preprocessing steps
    text = lower_case(text)
    text = remove_stop_words(text)
    text = removing_numbers(text)
    text = removing_punctuations(text)
    text = removing_urls(text)
    text = lemmatization(text)
    
    # Additional cleaning
    # Remove special characters and extra whitespace
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


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


# Set up MLflow tracking
dagshub.init(repo_owner='akhilcjdata', repo_name='YT-Capston-Project', mlflow=True)
mlflow.set_tracking_uri('https://dagshub.com/akhilcjdata/YT-Capston-Project.mlflow')

# Initialize Flask app
app = Flask(__name__)

# Create a custom registry for metrics
registry = CollectorRegistry()

# Define custom metrics
REQUEST_COUNT = Counter(
    "app_request_count", "Total number of requests to the app", ["method", "endpoint"], registry=registry
)
REQUEST_LATENCY = Histogram(
    "app_request_latency_seconds", "Latency of requests in seconds", ["endpoint"], registry=registry
)
PREDICTION_COUNT = Counter(
    "model_prediction_count", "Count of predictions for each class", ["prediction"], registry=registry
)

# Load tokenizer
tokenizer_data = pickle.load(open('models/bert_tokenizer_model_info.pkl', 'rb'))
print(f"Loaded tokenizer data type: {type(tokenizer_data)}")
print(f"Tokenizer data keys: {tokenizer_data.keys() if isinstance(tokenizer_data, dict) else 'Not a dict'}")

if isinstance(tokenizer_data, dict):
    for key, value in tokenizer_data.items():
        print(f"Key: {key}, Type: {type(value)}")

tokenizer_name = tokenizer_data.get('tokenizer_name', "bert-base-uncased")
model_name = tokenizer_data.get('model_name', "bert-base-uncased")

print(f"Loading tokenizer: {tokenizer_name}")
print(f"For model: {model_name}")

try:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    print(f"Successfully loaded tokenizer: {tokenizer.__class__.__name__}")
except Exception as e:
    print(f"Error loading tokenizer: {e}")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    print("Falling back to default bert-base-uncased tokenizer")

# Model setup
model_name = "my_model"
def get_latest_model_version(model_name):
    client = mlflow.MlflowClient()
    # Try to get the model from Staging stage
    latest_version = client.get_latest_versions(model_name, stages=["Staging"])
    if not latest_version:
        # Fallback to Production stage
        latest_version = client.get_latest_versions(model_name, stages=["Production"])
    if not latest_version:
        # Fallback to None stage
        latest_version = client.get_latest_versions(model_name, stages=["None"])
    return latest_version[0].version if latest_version else None

# Load MLflow model
try:
    model_version = get_latest_model_version(model_name)
    model_uri = f'models:/{model_name}/{model_version}'
    print(f"Fetching model from: {model_uri}")
    mlflow_model = mlflow.pyfunc.load_model(model_uri)
    print(f"Model loaded successfully. Type: {type(mlflow_model)}")
except Exception as e:
    print(f"Error loading model: {e}")
    raise

# Custom prediction function for BERT
def predict_sentiment(text):
    """
    Make a prediction using the BERT model
    """
    try:
        print(f"Processing text: {text[:50]}...")
        
        # Skip preprocessing for BERT (or apply minimal preprocessing)
        # text = normalize_text(text)
        
        # Tokenize
        inputs = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        
        # Convert to numpy for MLflow model
        input_dict = {k: v.numpy() for k, v in inputs.items()}
        
        # Create a dict where each token is a separate column
        # This is a direct approach that worked in many cases
        input_array = {}
        
        # Add each token as a separate column
        for i, token_id in enumerate(inputs['input_ids'][0].numpy()):
            input_array[f"input_ids_{i}"] = token_id
            
        # Add attention mask
        for i, mask_val in enumerate(inputs['attention_mask'][0].numpy()):
            input_array[f"attention_mask_{i}"] = mask_val
            
        # Add token type ids if present
        if 'token_type_ids' in inputs:
            for i, type_id in enumerate(inputs['token_type_ids'][0].numpy()):
                input_array[f"token_type_ids_{i}"] = type_id
        
        # Create DataFrame with a single row (for this prediction)
        input_df = pd.DataFrame([input_array])
        
        # Ensure all values are integers
        for col in input_df.columns:
            input_df[col] = input_df[col].astype('int32')
            
        print(f"Input DataFrame shape: {input_df.shape}")
        print(f"First few columns: {list(input_df.columns)[:5]}")
        
        # Get prediction from MLflow model
        try:
            # Attempt 1: Pass direct DataFrame
            prediction = mlflow_model.predict(input_df)
            print(f"Prediction successful. Result type: {type(prediction)}")
        except Exception as e1:
            print(f"Primary prediction method failed: {e1}")
            try:
                # Attempt 2: Create a PyTorch raw tensor
                prediction = torch.argmax(mlflow_model._model_impl.pytorch_model(**inputs).logits).item()
                print(f"Direct PyTorch prediction successful: {prediction}")
                return prediction
            except Exception as e2:
                print(f"Secondary prediction method failed: {e2}")
                raise
        
        # Process result
        if isinstance(prediction, np.ndarray):
            result = int(prediction[0])  # Get the first prediction
        elif isinstance(prediction, list):
            result = prediction[0]
        else:
            result = prediction
            
        return result
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        raise

# Routes
@app.route("/")
def home():
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start_time = time.time()
    response = render_template("index.html", result=None)
    REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
    return response

@app.route("/predict", methods=["POST"])
def predict():
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    start_time = time.time()
    
    try:
        text = request.form["text"]
        print(f"Received text for prediction: {text[:50]}...")
        
        # Get prediction
        prediction = predict_sentiment(text)
        
        # Map numerical prediction to sentiment for logging
        sentiment_text = "Positive" if prediction == 1 else "Negative"
        print(f"Prediction (raw): {prediction}")
        print(f"Sentiment: {sentiment_text}")
        
        # Increment prediction count metric
        PREDICTION_COUNT.labels(prediction=str(prediction)).inc()
        
        # Measure latency
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)
        
        # Return the numerical result to match the HTML template expectation
        # The template expects result to be 1 (positive) or 0 (negative)
        return render_template("index.html", result=prediction)
    
    except Exception as e:
        print(f"Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        
        # Still record latency even if there's an error
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - start_time)
        
        # Return error message that will show in the template
        return render_template("index.html", result=None, error=str(e))

@app.route("/metrics", methods=["GET"])
def metrics():
    """Expose Prometheus metrics."""
    return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}

if __name__ == "__main__":
    # app.run(debug=True) # for local use
    app.run(debug=True, host="0.0.0.0", port=5000)  # Accessible from outside Docker