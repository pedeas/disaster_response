import pandas as pd
from sqlalchemy import create_engine
import sys
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
import joblib
nltk.download('punkt')
nltk.download('wordnet')


def load_data(database_filepath):
    engine = create_engine(f'sqlite:///{database_filepath}')
    conn = engine.connect()
    df = pd.read_sql_table('DisasterResponse', conn)

    X = df['message']
    Y = df.iloc[:, 3:]
    category_names = Y.columns

    return X, Y, category_names


def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ]))
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # specify parameters for grid search
    parameters = {
       # 'clf__estimator__max_depth': [4, 16, 32]
        #'clf__estimator__min_samples_split': [2, 100],
        'clf__estimator__n_estimators': [10],
        'clf__estimator__criterion': ['gini', 'entropy'],
        'clf__estimator__max_features': [5, 100, 'auto']
    }

    # create grid search object
    cv = GridSearchCV(pipeline, cv=3, param_grid=parameters)

    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = pd.DataFrame(model.predict(X_test), columns=category_names)
    Y_pred = (Y_pred == 1)
    Y_test = (Y_test == 1)

    print('Evaluation of models per category:')
    for category in category_names:
        f1 = f1_score(Y_test[category], Y_pred[category], pos_label=1)
        recall = recall_score(Y_test[category], Y_pred[category], pos_label=1)
        precision = precision_score(Y_test[category], Y_pred[category], pos_label=1)
        print(f'Category: {category}')
        print(f'F1 = {round(f1, 2)}')
        print(f'Recall = {round(recall, 2)}')
        print(f'Precision = {round(precision, 2)}')
        print('----------------------------')
    pass


def save_model(model, model_filepath):
    joblib.dump(model, model_filepath)
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print(X_train.shape)
        print(Y_train.shape)
        
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