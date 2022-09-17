import sys
import re
import pickle
import numpy as np
import pandas as pd

# database
from sqlalchemy import create_engine

# pre processing
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords', 'omw-1.4'])

# modeling
from sklearn.base import ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


def load_data(database_filepath:str) -> tuple:
    """Reads data from database.

    Args:
        database_filepath (str): Filepath of the database.

    Returns:
        X (np.ndarray): Phrases to classify.
        Y (np.ndarray): Labels for the model.
        category_names: Category names list.
    """
    engine = create_engine(f'sqlite:///{database_filepath}')
    df = pd.read_sql_table(database_filepath, engine)
    category_names = df.iloc[:,4:].columns

    X = df[['message']]
    Y = df.iloc[:,4:]

    return X.values.ravel(), Y.values, category_names

def tokenize(text:str) -> list:
    """Tokenizes the input text.

    Args:
        text (str): Input text.

    Returns:
        clean_tokens: List of cleaned tokens.
    """
    text = re.sub(r"[^a-zA-Z0-9]", " ", text).lower()

    tokens = word_tokenize(text)
    lemm = WordNetLemmatizer()

    tokens = [w for w in tokens if w not in stopwords.words("english")]

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemm.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model() -> Pipeline:
    """Builds a model pipeline."""
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])

    return pipeline

def evaluate_model(
    model:Pipeline,
    X_test:np.ndarray,
    Y_test:np.ndarray,
    category_names:str
    ) -> None:
    """Evaluates the model.

    Args:
        model (Pipeline): Model pipeline.
        X_test (np.ndarray): Test dataset.
        Y_test (np.ndarray): Test labels.
        category_names (str): Category names list.
    """
    Y_pred = model.predict(X_test.ravel())
    Y_pred = pd.DataFrame(Y_pred, columns=category_names)
    Y_test = pd.DataFrame(Y_test, columns=category_names)
    for column in category_names:
        print(column)
        print(classification_report(Y_test[column], Y_pred[column]))

def save_model(model:ClassifierMixin, model_filepath:str) -> None:
    """Saves model as pickle.

    Args:
        model (ClassifierMixin): Model pipeline to be saved.
        model_filepath (str): Model filepath.
    """
    pickle.dump(model, open(model_filepath, 'wb'))

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print(f'Loading data...\n    DATABASE: {database_filepath}')
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print(f'Saving model...\n    MODEL: {model_filepath}')
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()
