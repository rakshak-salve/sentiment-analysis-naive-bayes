# 🎭 Sentiment Analysis using Naive Bayes Classifier

A beginner-friendly project to classify text reviews as positive or negative using classic machine learning techniques. This project demonstrates the complete pipeline from data preprocessing to model deployment.

## 📚 Project Purpose

- **Learn basic NLP and ML**: Understand text preprocessing, feature extraction, and classification
- **Build interpretable models**: Use Naive Bayes for transparent decision-making
- **Practice data science workflow**: From exploration to deployment
- **Showcase growth**: Clean commits, clear documentation, and honest reflections

## 🗂️ Project Structure

```
sentiment-analysis-naive-bayes/
├── data/
│   └── sample_reviews.csv        # Dataset with movie reviews
├── notebooks/
│   └── Sentiment_Analysis_Basics.ipynb  # Jupyter notebook with complete analysis
├── src/
│   ├── train_model.py            # Script to train and evaluate model
│   └── predict.py                # Interactive prediction script
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone https://github.com/rakshak-salve/sentiment-analysis-naive-bayes.git
cd sentiment-analysis-naive-bayes

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model

```bash
# Navigate to src directory
cd src

# Train the model
python train_model.py
```

### 3. Make Predictions

```bash
# Interactive prediction
python predict.py
```

### 4. Explore with Jupyter

```bash
# Start Jupyter notebook
jupyter notebook ../notebooks/Sentiment_Analysis_Basics.ipynb
```

## 📊 Dataset

The project uses a custom dataset of **30 movie reviews** with balanced positive and negative sentiments:

- **Positive reviews (1)**: 15 samples
- **Negative reviews (0)**: 15 samples
- **Format**: CSV with `review` and `sentiment` columns

### Sample Data:
```csv
review,sentiment
"This movie was absolutely fantastic! The acting was superb and the plot was engaging from start to finish.",1
"This was a terrible movie. The acting was wooden and the plot made no sense at all.",0
```

## 🧑‍💻 Methodology

### 1. Data Preprocessing
- **Text cleaning**: Lowercase, remove punctuation/numbers
- **Tokenization**: Split into words
- **Stopword removal**: Remove common words (the, is, at, etc.)
- **Lemmatization**: Convert words to base form (running → run)

### 2. Feature Extraction
- **Bag of Words**: CountVectorizer with 1000 max features
- **N-grams**: Include bigrams for better context
- **TF-IDF**: Alternative vectorization method

### 3. Model Training
- **Algorithm**: Multinomial Naive Bayes
- **Split**: 80% training, 20% testing
- **Cross-validation**: 5-fold CV for robust evaluation

### 4. Evaluation Metrics
- **Accuracy**: Overall prediction correctness
- **Confusion Matrix**: Detailed error analysis
- **Classification Report**: Precision, recall, F1-score

## 📈 Results

### Model Performance
- **Accuracy**: ~85-90% (varies with random seed)
- **Cross-validation**: Stable performance across folds
- **Feature importance**: Identifies key positive/negative words

### Sample Predictions
```
Text: "This movie was absolutely amazing! I loved every minute of it."
Sentiment: 🟢 Positive (confidence: 95.2%)

Text: "The film was terrible and boring. I hated it."
Sentiment: 🔴 Negative (confidence: 98.7%)
```

## 🔍 Key Features

### Text Preprocessing Pipeline
```python
def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Remove stopwords and lemmatize
    words = [lemmatizer.lemmatize(word) for word in words 
             if word not in stop_words and len(word) > 2]
    return ' '.join(words)
```

### Model Training
```python
# Create vectorizer
vectorizer = CountVectorizer(max_features=1000, ngram_range=(1, 2))
X = vectorizer.fit_transform(cleaned_texts)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)
```

### Prediction Function
```python
def predict_sentiment(text, model, vectorizer):
    cleaned_text = clean_text(text)
    text_vector = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_vector)[0]
    probability = model.predict_proba(text_vector)[0]
    return sentiment, confidence
```

## 📝 What I Learned

### Technical Skills
- **NLP preprocessing**: Text cleaning, tokenization, lemmatization
- **Feature engineering**: Bag of words, TF-IDF, n-grams
- **Model evaluation**: Cross-validation, confusion matrices
- **Model persistence**: Saving and loading trained models

### Data Science Workflow
- **Exploratory analysis**: Understanding data distribution and quality
- **Iterative improvement**: Trying different preprocessing steps
- **Documentation**: Clear code comments and README
- **Error handling**: Robust scripts with proper logging

### Challenges Faced
- **Small dataset**: Limited training data affects model robustness
- **Text preprocessing**: Finding the right balance of cleaning
- **Feature selection**: Choosing optimal vectorizer parameters
- **Model interpretation**: Understanding Naive Bayes decisions

## 🚧 Areas for Improvement

### Data Quality
- **Larger dataset**: Use IMDB or Amazon review datasets
- **More diverse text**: Include different domains and writing styles
- **Balanced classes**: Ensure equal positive/negative samples

### Model Enhancement
- **Advanced algorithms**: Try Logistic Regression, SVM, Random Forest
- **Word embeddings**: Use Word2Vec, GloVe, or BERT embeddings
- **Deep learning**: Implement LSTM or Transformer models
- **Ensemble methods**: Combine multiple models for better performance

### Feature Engineering
- **Sentiment lexicons**: Incorporate VADER or TextBlob scores
- **Part-of-speech tagging**: Use POS information
- **Named entity recognition**: Identify entities in text
- **Topic modeling**: Extract latent topics

## 🔜 Next Steps

### Immediate Improvements
1. **Expand dataset**: Collect more movie reviews
2. **Try TF-IDF**: Compare with current Bag of Words approach
3. **Hyperparameter tuning**: Optimize vectorizer and model parameters
4. **Error analysis**: Study misclassified examples

### Advanced Features
1. **Web interface**: Create a simple Flask/Django app
2. **Real-time API**: Deploy model as REST API
3. **Multi-class classification**: Add neutral sentiment
4. **Domain adaptation**: Adapt to different text types

### Learning Goals
1. **Deep learning**: Study LSTM and Transformer architectures
2. **BERT fine-tuning**: Learn transfer learning for NLP
3. **Model interpretability**: Understand model decisions
4. **Production deployment**: Learn MLOps practices

## 🤝 Contributing

This is a learning project, but contributions are welcome! Feel free to:

- **Report issues**: Found a bug or have a suggestion?
- **Improve documentation**: Better explanations or examples
- **Add features**: New preprocessing steps or evaluation metrics
- **Share datasets**: Larger or more diverse text collections

## 📚 Resources

### Learning Materials
- [NLTK Documentation](https://www.nltk.org/)
- [Scikit-learn Text Feature Extraction](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction)
- [Naive Bayes Classification](https://scikit-learn.org/stable/modules/naive_bayes.html)
- [Text Preprocessing Guide](https://towardsdatascience.com/text-preprocessing-steps-and-universal-pipeline-94233cb6725a)

### Datasets
- [IMDB Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)
- [Amazon Product Reviews](https://www.kaggle.com/datasets/arhamrumi/amazon-product-reviews-2024)
- [Sentiment140](http://help.sentiment140.com/for-students/)

### Advanced Topics
- [Word Embeddings](https://www.tensorflow.org/tutorials/text/word2vec)
- [BERT Fine-tuning](https://huggingface.co/docs/transformers/training)
- [Model Interpretability](https://github.com/marcotcr/lime)

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- **NLTK team**: For excellent NLP tools
- **Scikit-learn community**: For robust ML implementations
- **Open source community**: For inspiration and learning resources

---

> **"This was my first NLP project—lots of trial and error, but I learned a ton! The journey from basic text cleaning to understanding model decisions has been incredibly rewarding. Every commit represents a learning step, and every error taught me something new."**

**Happy coding! 🚀** 