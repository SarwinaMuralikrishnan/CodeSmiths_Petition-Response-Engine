import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

train_df = pd.read_excel("Train Data.xlsx").dropna(subset=["description", "department"])
test_df = pd.read_excel("Test Data.xlsx").dropna(subset=["description", "department"])

X_train = train_df["description"].astype(str)
y_train = train_df["department"].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)

X_test = test_df["description"].astype(str)
y_test = test_df["department"].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)

vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

with open("analysis.txt", "w") as f:
    f.write(f"Accuracy: {accuracy_score(y_test, y_pred)}\n")
    f.write(f"Train samples: {len(X_train)}\n")
    f.write(f"Test samples: {len(X_test)}\n")
    f.write(f"Train classes: {len(y_train.unique())}\n")
    f.write(f"Test classes: {len(y_test.unique())}\n")
    
    missing_classes = set(y_test.unique()) - set(y_train.unique())
    f.write(f"Missing classes in train: {missing_classes}\n")
    
    f.write("Test Report:\n")
    f.write(classification_report(y_test, y_pred))
