import sys

# download necessary NLTK data
import nltk
nltk.download(['punkt', 'wordnet'])

# import libraries
import numpy as np
import pandas as pd
import pickle

from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier


def load_data(database_filepath):
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Messages', engine)
    df = df[df.message.notnull()]
    df = df.dropna()

    # Build the X, y and category_names variables
    X = df['message']
    y = df.drop(['id','message','original','genre'], axis=1)
    category_names = list(df.columns)

    return X, y, category_names


def tokenize(text):
    # tokenize text
    tokens = word_tokenize(text)

    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer()

    # iterate through each token
    clean_tokens = []
    for tok in tokens:
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    # Build the machine learning pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    return pipeline


def evaluate_model(model, X_test, y_test, category_names):
    # predict on test data
    y_pred = model.predict(X_test)

    # test the model
    col_idx = 0
    for col_name in y_test:
        print('column: ', col_name)
        print(classification_report(y_test[col_name], y_pred[:,col_idx]))
        col_idx = col_idx + 1


def save_model(model, model_filepath):
    # Save to file in the current working directory
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
