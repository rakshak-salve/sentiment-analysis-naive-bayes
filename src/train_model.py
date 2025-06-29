#!/usr/bin/env python3
"""
Sentiment Analysis Model Training Script

This script trains a Naive Bayes classifier for sentiment analysis
and saves the trained model and vectorizer for later use.
"""

import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
import joblib
import logging
import os
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def download_nltk_data():
    """Download required NLTK data"""
    try:
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        nltk.download('punkt', quiet=True)
        logger.info("NLTK data downloaded successfully")
    except Exception as e:
        logger.error(f"Failed to download NLTK data: {e}")
        raise

def clean_text(text):
    """
    Clean and preprocess text data
    
    Args:
        text (str): Input text to clean
        
    Returns:
        str: Cleaned text
    """
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    
    # Split into words
    words = text.split()
    
    # Remove stopwords and lemmatize
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    words = [lemmatizer.lemmatize(word) for word in words 
             if word not in stop_words and len(word) > 2]
    
    # Join words back into text
    return ' '.join(words)

def load_and_preprocess_data(data_path):
    """
    Load and preprocess the dataset
    
    Args:
        data_path (str): Path to the CSV file
        
    Returns:
        tuple: (X, y) where X is the feature matrix and y is the target
    """
    logger.info(f"Loading data from {data_path}")
    
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Loaded {len(df)} reviews")
        
        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            logger.warning(f"Found missing values: {missing_values}")
            df = df.dropna()
        
        # Check class distribution
        sentiment_counts = df['sentiment'].value_counts()
        logger.info(f"Class distribution: {sentiment_counts.to_dict()}")
        
        # Clean text
        logger.info("Cleaning text data...")
        df['clean_review'] = df['review'].apply(clean_text)
        
        # Remove empty reviews after cleaning
        df = df[df['clean_review'].str.len() > 0]
        logger.info(f"After cleaning: {len(df)} reviews remaining")
        
        return df['clean_review'], df['sentiment']
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def create_vectorizer():
    """
    Create and configure the text vectorizer
    
    Returns:
        CountVectorizer: Configured vectorizer
    """
    vectorizer = CountVectorizer(
        max_features=1000,
        min_df=2,
        max_df=0.8,
        ngram_range=(1, 2),  # Include bigrams
        stop_words='english'
    )
    return vectorizer

def train_model(X, y, test_size=0.2, random_state=42):
    """
    Train the sentiment analysis model
    
    Args:
        X: Feature matrix
        y: Target labels
        test_size: Proportion of data for testing
        random_state: Random seed for reproducibility
        
    Returns:
        tuple: (model, vectorizer, X_test, y_test, y_pred)
    """
    logger.info("Training sentiment analysis model...")
    
    # Create vectorizer
    vectorizer = create_vectorizer()
    
    # Transform text to features
    X_vectorized = vectorizer.fit_transform(X)
    logger.info(f"Feature matrix shape: {X_vectorized.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logger.info(f"Training set size: {X_train.shape[0]}")
    logger.info(f"Test set size: {X_test.shape[0]}")
    
    # Train model
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    logger.info("Model training completed")
    
    return model, vectorizer, X_test, y_test, y_pred

def evaluate_model(y_test, y_pred, model, X_vectorized, y):
    """
    Evaluate the trained model
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        model: Trained model
        X_vectorized: Vectorized feature matrix
        y: Original target labels
    """
    logger.info("Evaluating model performance...")
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    logger.info(f"Accuracy: {accuracy:.3f}")
    logger.info(f"Confusion Matrix:\n{conf_matrix}")
    
    # Classification report
    report = classification_report(y_test, y_pred, target_names=['Negative', 'Positive'])
    logger.info(f"Classification Report:\n{report}")
    
    # Cross-validation score
    cv_scores = cross_val_score(model, X_vectorized, y, cv=5)
    logger.info(f"Cross-validation scores: {cv_scores}")
    logger.info(f"CV Mean: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    return accuracy, conf_matrix

def save_model(model, vectorizer, model_path, vectorizer_path):
    """
    Save the trained model and vectorizer
    
    Args:
        model: Trained model
        vectorizer: Fitted vectorizer
        model_path: Path to save the model
        vectorizer_path: Path to save the vectorizer
    """
    try:
        joblib.dump(model, model_path)
        joblib.dump(vectorizer, vectorizer_path)
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Vectorizer saved to {vectorizer_path}")
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise

def predict_sentiment(text, model, vectorizer):
    """
    Predict sentiment for a given text
    
    Args:
        text (str): Input text
        model: Trained model
        vectorizer: Fitted vectorizer
        
    Returns:
        dict: Prediction results
    """
    # Clean the text
    cleaned_text = clean_text(text)
    
    # Vectorize the text
    text_vector = vectorizer.transform([cleaned_text])
    
    # Make prediction
    prediction = model.predict(text_vector)[0]
    probability = model.predict_proba(text_vector)[0]
    
    sentiment = "Positive" if prediction == 1 else "Negative"
    confidence = probability[1] if prediction == 1 else probability[0]
    
    return {
        'text': text,
        'cleaned_text': cleaned_text,
        'sentiment': sentiment,
        'confidence': confidence,
        'probabilities': {
            'negative': probability[0],
            'positive': probability[1]
        }
    }

def main():
    """Main function to run the training pipeline"""
    logger.info("Starting sentiment analysis model training")
    
    # Download NLTK data
    download_nltk_data()
    
    # File paths
    data_path = "../data/sample_reviews.csv"
    model_path = "sentiment_model.pkl"
    vectorizer_path = "vectorizer.pkl"
    
    try:
        # Load and preprocess data
        X, y = load_and_preprocess_data(data_path)
        
        # Train model
        model, vectorizer, X_test, y_test, y_pred = train_model(X, y)
        
        # Evaluate model
        accuracy, conf_matrix = evaluate_model(y_test, y_pred, model, vectorizer.transform(X), y)
        
        # Save model
        save_model(model, vectorizer, model_path, vectorizer_path)
        
        # Test with some examples
        test_texts = [
            "This movie was absolutely amazing! I loved every minute of it.",
            "The film was terrible and boring. I hated it.",
            "The movie was okay, nothing special.",
            "Incredible acting and beautiful cinematography made this film unforgettable.",
            "Poor direction and terrible acting ruined this movie completely."
        ]
        
        logger.info("Testing model with sample texts:")
        for text in test_texts:
            result = predict_sentiment(text, model, vectorizer)
            logger.info(f"Text: {result['text']}")
            logger.info(f"Sentiment: {result['sentiment']} (confidence: {result['confidence']:.3f})")
            logger.info(f"Probabilities - Negative: {result['probabilities']['negative']:.3f}, "
                       f"Positive: {result['probabilities']['positive']:.3f}")
            logger.info("-" * 50)
        
        logger.info("Training pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main() 