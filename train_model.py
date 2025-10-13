# train_model.py (Improved Spam/Ham Classifier)
import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------
# 1. Load dataset
# -------------------------------
data = pd.read_csv("spam.csv", encoding='latin-1')

# Clean extra columns
data = data.rename(columns={"v1": "label", "v2": "message"})
data = data[["label", "message"]]

# -------------------------------
# 2. Text cleaning function
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # remove links
    text = re.sub(r'\d+', '', text)                                          # remove numbers
    text = re.sub(r'[^a-z\s]', '', text)                                     # remove special chars
    text = re.sub(r'\s+', ' ', text).strip()                                 # remove extra spaces
    return text

data['clean_message'] = data['message'].apply(clean_text)

# Encode labels (ham = 0, spam = 1)
data['label_num'] = data['label'].map({'ham': 0, 'spam': 1})

# -------------------------------
# 3. Train/test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    data['clean_message'], data['label_num'], test_size=0.2, random_state=42
)

# -------------------------------
# 4. Build improved model
# -------------------------------
model = Pipeline([
    ('tfidf', TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),     # capture "won lottery", "account number"
        max_df=0.9,             # ignore extremely frequent words
        min_df=2                # ignore rare words
    )),
    ('lr', LogisticRegression(max_iter=1000))
])

# -------------------------------
# 5. Train
# -------------------------------
model.fit(X_train, y_train)

# -------------------------------
# 6. Evaluate
# -------------------------------
pred = model.predict(X_test)
print("✅ Model Accuracy:", round(accuracy_score(y_test, pred)*100, 2), "%")
print("\nClassification Report:\n", classification_report(y_test, pred))

# -------------------------------
# 7. Save model
# -------------------------------
joblib.dump(model, 'spam_model.pkl')
print("🎉 Improved model saved as spam_model.pkl")
