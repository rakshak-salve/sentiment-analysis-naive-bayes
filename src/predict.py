#!/usr/bin/env python3
"""
Simple Sentiment Prediction Script

This script loads a trained sentiment analysis model and allows users
to input text for sentiment prediction.
"""

import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd

def clean_text(text):
    """
    Clean and preprocess text data
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

def predict_sentiment(text, model, vectorizer):
    """
    Predict sentiment for a given text
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
    """Main function for interactive prediction"""
    print("=" * 60)
    print("🎭 Sentiment Analysis Predictor")
    print("=" * 60)
    
    try:
        # Load the trained model and vectorizer
        print("Loading model...")
        model = joblib.load('sentiment_model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
        print("✅ Model loaded successfully!")
        
    except FileNotFoundError:
        print("❌ Error: Model files not found!")
        print("Please run the training script first: python train_model.py")
        return
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    print("\nEnter text to analyze sentiment (type 'quit' to exit):")
    print("-" * 60)
    
    while True:
        try:
            # Get user input
            text = input("\n📝 Enter text: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not text:
                print("⚠️  Please enter some text.")
                continue
            
            # Make prediction
            result = predict_sentiment(text, model, vectorizer)
            
            # Display results
            print("\n" + "=" * 40)
            print("📊 PREDICTION RESULTS")
            print("=" * 40)
            print(f"Original text: {result['text']}")
            print(f"Cleaned text: {result['cleaned_text']}")
            print(f"Sentiment: {'🟢' if result['sentiment'] == 'Positive' else '🔴'} {result['sentiment']}")
            print(f"Confidence: {result['confidence']:.1%}")
            print(f"Probabilities:")
            print(f"  - Negative: {result['probabilities']['negative']:.1%}")
            print(f"  - Positive: {result['probabilities']['positive']:.1%}")
            print("=" * 40)
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main() 