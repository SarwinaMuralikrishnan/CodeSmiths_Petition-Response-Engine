import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

train_df = pd.read_excel("Train Data.xlsx")
test_df = pd.read_excel("Test Data.xlsx")

X_train = train_df["description"]
y_train = train_df["department"]

train_df = train_df.dropna(subset=["description", "department"])

X_train = train_df["description"].astype(str)
y_train = train_df["department"].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)

vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

pickle.dump(model, open("petition_model.pkl", "wb"))
pickle.dump(vectorizer, open("tfidf.pkl", "wb"))

print("Model and vectorizer saved successfully")
