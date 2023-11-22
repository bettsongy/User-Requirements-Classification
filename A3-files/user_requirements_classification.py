import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from scipy.sparse import hstack, csr_matrix
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import nltk
import re
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline  # Corrected import


# Ensure nltk resources are downloaded
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

import re

# Load data


def load_data(paths):
    dataframes = []
    for path in paths:
        df = pd.read_json(path)
        dataframes.append(df)
    return pd.concat(dataframes, ignore_index=True)


def train_model_with_hyperparameter_tuning(X_train, y_train):
    # Define the parameter grid for RandomForest
    param_grid_rf = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    rf = RandomForestClassifier()
    grid_search_rf = GridSearchCV(
        estimator=rf, param_grid=param_grid_rf, cv=3, n_jobs=-1, scoring='accuracy')
    grid_search_rf.fit(X_train, y_train)
    return grid_search_rf.best_estimator_

# Preprocess text

def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    # Text normalization and tokenization
    text = re.sub(r'[^a-zA-Z0-9]', ' ', str(text).lower())
    tokens = nltk.word_tokenize(text)
    # Lemmatization and stopword removal
    processed_text = ' '.join([lemmatizer.lemmatize(word)
                               for word in tokens if word not in stop_words and len(word) > 1])
    return processed_text


def preprocess_data(data):
    # Handle NaN values in text columns
    data['comment'] = data['comment'].fillna('')
    data['title'] = data['title'].fillna('')

    # Apply text preprocessing
    data['processed_text'] = data.apply(lambda row: preprocess_text(
        row['comment'] + ' ' + row['title']), axis=1)

    # Replace NaN with 0 before scaling
    data['sentiScore'] = data['sentiScore'].fillna(0)
    data['rating'] = data['rating'].fillna(0)

    # Apply MinMaxScaler
    scaler = MinMaxScaler()
    data[['sentiScore_norm', 'rating_norm']] = scaler.fit_transform(
        data[['sentiScore', 'rating']])

    return data

# Feature extraction
def feature_extraction(data):
    vectorizer = TfidfVectorizer(max_features=1000)
    text_features = vectorizer.fit_transform(data['processed_text'])

    # Convert the DataFrame to a sparse matrix
    additional_features = csr_matrix(
        data[['sentiScore_norm', 'rating_norm']].to_numpy())

    # Combine with text features
    combined_features = hstack([text_features, additional_features])
    return combined_features
# Train and evaluate models


def train_and_evaluate(data, labels):
    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2)

    # Define hyperparameter grids
    rf_grid = {
        'randomforestclassifier__n_estimators': [100, 200, 300, 500],
        'randomforestclassifier__max_depth': [10, 20, 30, None],
        'randomforestclassifier__min_samples_split': [2, 5, 10],
        'randomforestclassifier__min_samples_leaf': [1, 2, 4],
        'randomforestclassifier__bootstrap': [True, False]
    }
    svm_grid = {
        'svc__C': [0.1, 1, 10, 100],
        'svc__gamma': ['scale', 'auto', 0.1, 1, 10],
        'svc__kernel': ['rbf', 'poly', 'sigmoid']
    }
    nb_grid = {
        'multinomialnb__alpha': [1.0, 0.5, 0.1, 0.01, 0.001]
    }
    dt_grid = {
        'decisiontreeclassifier__max_depth': [5, 10, 15, 20, None],
        'decisiontreeclassifier__min_samples_split': [2, 5, 10],
        'decisiontreeclassifier__min_samples_leaf': [1, 2, 4],
        'decisiontreeclassifier__criterion': ['gini', 'entropy']
    }

    # Models with their corresponding hyperparameter grids

    models = {
        'RandomForest': (RandomForestClassifier(), rf_grid),
        'SVM': (SVC(), svm_grid),
        'NaiveBayes': (MultinomialNB(), nb_grid),
        'DecisionTree': (DecisionTreeClassifier(), dt_grid)
    }

    for name, (model, grid) in models.items():
        print(f"Training and evaluating {name}")

        # Define the pipeline with named steps
        model_pipeline = make_pipeline(
            SMOTE(),
            model
        )

        grid_search = GridSearchCV(
            estimator=model_pipeline, param_grid=grid, cv=3, n_jobs=-1, scoring='accuracy')
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        print(f"Results for {name} with Hyperparameter Tuning:")
        print(classification_report(y_test, y_pred))
        print("="*30)



# Paths to JSON files
file_paths = [
    'Bug_tt.json',
    'Feature_tt.json',
    'Rating_tt.json',
    'UserExperience_tt.json'
]


# Main execution
data = load_data(file_paths)
data = preprocess_data(data)
features = feature_extraction(data)
train_and_evaluate(features, data['label'])
