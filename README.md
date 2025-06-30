# Sentiment Analysis with Naive Bayes

Hi! This is a simple project to check if a movie review is happy (positive) or not happy (negative). I used Python and some easy machine learning tools.

## Why I Made This

- I want to learn how computers understand words.
- I want to try machine learning.
- I want to practice Python.

## What’s Inside

```
sentiment-analysis-naive-bayes/
├── data/                  # Movie reviews
├── notebooks/             # Jupyter notebook (not finished)
├── src/                   # Python code
├── requirements.txt       # What you need to install
└── README.md              # This file
```

## How To Use

1. **Get the code and install things**
    ```bash
    git clone https://github.com/rakshak-salve/sentiment-analysis-naive-bayes.git
    cd sentiment-analysis-naive-bayes
    pip install -r requirements.txt
    ```

2. **Train the model**
    ```bash
    cd src
    python train_model.py
    ```

3. **Test your own review**
    ```bash
    python predict.py
    ```

4. **(Optional) Open notebook**
    ```bash
    jupyter notebook ../notebooks/Sentiment_Analysis_Basics.ipynb
    ```

## About the Data

- 30 movie reviews
- Half are happy (1), half are not happy (0)
- Each row has a review and a number (1 or 0)

Example:
```csv
review,sentiment
"This movie was great!",1
"This movie was bad.",0
```

## How It Works

1. Clean the words (make small letters, remove weird stuff)
2. Change words to numbers
3. Train the model to learn
4. Test if it works

## What I Learned

- How to clean words
- How to use simple machine learning
- How to test if my code works

## What Can Be Better

- More reviews would help
- Try other models
- Maybe make a website

---

*This is just for learning. If you have ideas, tell