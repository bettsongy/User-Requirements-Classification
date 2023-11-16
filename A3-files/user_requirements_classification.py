import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import SGDClassifier
import nltk
nltk.download('wordnet')
nltk.download('stopwords')

from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re

def load_and_label_data(file_paths, labels):
    dataframes = []
    for path, label in zip(file_paths, labels):
        df = pd.read_json(path)
        df['label'] = label
        # Split into target and complement classes if necessary
        df_target = df[df['label'] == label]
        df_complement = df[df['label'] != label]
        # Rename complement label
        df_complement['label'] = 'Not_' + label
        dataframes.extend([df_target, df_complement])
    return pd.concat(dataframes, ignore_index=True)

def preprocess_text(text):
    # Custom lemmatization
    lemmatizer = WordNetLemmatizer()
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    
    # Stopword removal
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in lemmatized_words if word not in stop_words]
    
    # Rejoin words
    return ' '.join(filtered_words)


def preprocess_data(data):
    # Convert columns to object type before filling NaN values
    data = data.astype(object).fillna('')
    # Combine 'comment' with other fields and preprocess
    data['combined_text'] = data.apply(lambda row: preprocess_text(row['comment'] + ' ' + (row['title'] if pd.notnull(row['title']) else '')), axis=1)
    return data

def feature_extraction(data, column='combined_text', max_features=1000):
    # Using CountVectorizer for Word Count Vectors
    vectorizer = CountVectorizer(max_features=max_features, ngram_range=(1,2))
    features = vectorizer.fit_transform(data[column])
    # Normalize the feature matrix
    features = normalize(features)
    return features

def train_and_evaluate_model(X, y, model, params=None, cv=5):
    # Addressing data imbalance
    smote = SMOTE()
    X_res, y_res = smote.fit_resample(X, y)

    # Hyperparameter tuning using GridSearchCV if parameters are provided
    if params:
        model = GridSearchCV(model, params, cv=cv, scoring='accuracy')
    
    # Cross-validation
    scores = cross_val_score(model, X_res, y_res, cv=cv)
    print(f"Cross-Validation Scores: {scores}")
    print(f"Mean CV Score: {scores.mean()}")

    # Splitting the data for final evaluation
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2)

    # Train the classifier
    model.fit(X_train, y_train)

    # Predictions and final evaluation
    y_pred = model.predict(X_test)
    # Print classification report and confusion matrix
    print(f"Classification Report for {type(model).__name__}:\n")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))
    print("\n" + "="*80 + "\n")

# Paths to your JSON files and labels
file_paths = ['bug_tt.json', 'feature_tt.json', 'rating_tt.json', 'UserExperience_tt.json']
labels = ['Bug', 'Feature', 'Rating', 'UserExperience']

# Load and preprocess data
data = load_and_label_data(file_paths, labels)
data = preprocess_data(data)

# Define models and hyperparameters
models = {
    'RandomForest': (RandomForestClassifier(), {'n_estimators': [100, 200], 'max_depth': [10, 20]}),
    'NaiveBayes': (MultinomialNB(), {'alpha': [1.0, 0.5, 0.1]}),
    'SVM': (SVC(), {'C': [1, 10], 'kernel': ['linear', 'rbf']}),
    'DecisionTree': (DecisionTreeClassifier(), {'max_depth': [5, 10, 15]}),
    'SGD': (SGDClassifier(), {'loss': ['hinge', 'log'], 'alpha': [0.0001, 0.001], 'penalty': ['l2', 'l1']})
}

# Train and evaluate each model
for model_name, (model, params) in models.items():
    print(f"Training and evaluating {model_name}")
    X = feature_extraction(data)
    train_and_evaluate_model(X, data['label'], model, params)
