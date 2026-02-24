import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

train_df = pd.read_excel("Train Data.xlsx")
test_df = pd.read_excel("Test Data.xlsx")

train_df = train_df.dropna(subset=["description", "department"])
test_df = test_df.dropna(subset=["description", "department"])

X_train = train_df["description"].astype(str)
y_train = train_df["department"].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)

X_test = test_df["description"].astype(str)
y_test = test_df["department"].astype(str).str.strip().str.replace(r'\s+', ' ', regex=True)

vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

