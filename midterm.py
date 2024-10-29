import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from scipy.sparse import hstack

print("STARTING")

train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

train_data_with_score = train_data[train_data['Score'].notna()].copy()
train_data_with_score['HelpfulnessNumerator'] = train_data_with_score['HelpfulnessNumerator'].fillna(0)
train_data_with_score['HelpfulnessDenominator'] = train_data_with_score['HelpfulnessDenominator'].fillna(0)
train_data_with_score['Summary'] = train_data_with_score['Summary'].fillna('')
train_data_with_score['Text'] = train_data_with_score['Text'].fillna('')

train_data_with_score.dropna(subset=['ProductId', 'UserId'], inplace=True)

def extract_features(df):
    df['ReviewLength'] = df['Text'].apply(lambda x: len(x.split()))
    df['SummaryLength'] = df['Summary'].apply(lambda x: len(x.split()))
    df['ReviewCharLength'] = df['Text'].apply(len)
    df['SummaryCharLength'] = df['Summary'].apply(len)
    df['HelpfulnessRatio'] = df['HelpfulnessNumerator'] / (df['HelpfulnessDenominator'] + 1)
    return df[['ReviewLength', 'SummaryLength', 'ReviewCharLength', 'SummaryCharLength', 'HelpfulnessRatio']]

tfidf = TfidfVectorizer(max_features=2000, stop_words='english', ngram_range=(1, 1))
X_text = tfidf.fit_transform(train_data_with_score['Text'])

X_numeric = extract_features(train_data_with_score).values
X = hstack([X_text, X_numeric]).tocsr()

y = train_data_with_score['Score'].astype(int)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model_pipeline = Pipeline([
    ('scaler', StandardScaler(with_mean=False)),  
    ('classifier', LogisticRegression(C=0.3, max_iter=100, random_state=42, n_jobs=-1))
])

model_pipeline.fit(X_train, y_train)

y_val_pred = model_pipeline.predict(X_val)
accuracy = accuracy_score(y_val, y_val_pred)
print(f'Validation Accuracy: {accuracy:.4f}')

test_ids = test_data['Id']
test_data_matched = train_data[train_data['Id'].isin(test_ids)].copy()

test_data_matched['HelpfulnessNumerator'] = test_data_matched['HelpfulnessNumerator'].fillna(0)
test_data_matched['HelpfulnessDenominator'] = test_data_matched['HelpfulnessDenominator'].fillna(0)
test_data_matched['Summary'] = test_data_matched['Summary'].fillna('')
test_data_matched['Text'] = test_data_matched['Text'].fillna('')

X_test_text = tfidf.transform(test_data_matched['Text'])
X_test_numeric = extract_features(test_data_matched).values
X_test = hstack([X_test_text, X_test_numeric]).tocsr()

test_data_matched['Score'] = model_pipeline.predict(X_test)

submission = test_data[['Id']].merge(test_data_matched[['Id', 'Score']], on='Id', how='left')
submission.to_csv('submission.csv', index=False)
print("FILE CREATED")
