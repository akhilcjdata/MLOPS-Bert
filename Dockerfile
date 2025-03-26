FROM python:3.10-slim
WORKDIR /app
COPY flask_app/ /app/
COPY models/bert_tokenizer_model_info.pkl /app/models/bert_tokenizer_model_info.pkl
RUN pip install -r requirements.txt
# Replace NLTK downloads with spaCy model download
RUN python -m spacy download en_core_web_sm
EXPOSE 5000
#local
CMD ["python", "app.py"]  
#Prod
#CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--timeout", "120", "app:app"]
