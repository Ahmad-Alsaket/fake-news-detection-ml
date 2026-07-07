
import re
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# 1) Load the dataset
# ============================================
fake = pd.read_csv(r"C:\Users\USER\Documents\vspython\Fake.csv")
real = pd.read_csv(r"C:\Users\USER\Documents\vspython\True.csv")

# Add labels:
fake["label"] = 0
real["label"] = 1

# Merge both datasets into one dataframe
data = pd.concat([fake, real], ignore_index=True)



# 2) Create a single text column (content)
#    Combine title + text 
# ============================================
data["content"] = data["title"].fillna("") + " " + data["text"].fillna("")
data = data[["content", "label"]]

# Shuffle the dataset so fake/real are mixed
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# ============================================
# 3) Text Cleaning (basic preprocessing)
# ============================================
def clean_text(s: str) -> str:
    s = s.lower()                                 # lowercase
    s = re.sub(r"http\S+|www\S+", " ", s)          # remove links
    s = re.sub(r"[^a-z\s]", " ", s)                # keep only letters and spaces
    s = re.sub(r"\s+", " ", s).strip()             # remove extra spaces
    return s

data["clean_content"] = data["content"].apply(clean_text)

# ============================================
# 4) Train/Test Split
#    X = input text , y = labels
# ============================================
X = data["clean_content"]
y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train size:", X_train.shape[0])
print("Test size:", X_test.shape[0])

# ============================================
# 5) Convert Text -> Numbers using TF-IDF
#    Fit on training only (avoid data leakage)

# ============================================
tfidf = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("Train TF-IDF shape:", X_train_tfidf.shape)
print("Test TF-IDF shape:", X_test_tfidf.shape)

# ============================================
# 6) Model 1: Logistic Regression
# ============================================
model_lr = LogisticRegression(max_iter=1000)
model_lr.fit(X_train_tfidf, y_train)

y_pred_lr = model_lr.predict(X_test_tfidf)

print("\n--- Logistic Regression ---")
print("Accuracy:", accuracy_score(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))

# ============================================
# 7) Model 2: Naive Bayes (MultinomialNB)
# ============================================
model_nb = MultinomialNB()
model_nb.fit(X_train_tfidf, y_train)

y_pred_nb = model_nb.predict(X_test_tfidf)

print("\n--- Naive Bayes (MultinomialNB) ---")
print("Accuracy:", accuracy_score(y_test, y_pred_nb))
print(classification_report(y_test, y_pred_nb))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_nb))

# ============================================
# 8) EDA Visualizations
# ============================================

# 8.1 Class distribution (Fake vs Real)
data["label"].value_counts().plot(kind="bar")
plt.xticks([0, 1], ["Fake", "Real"], rotation=0)
plt.title("Distribution of Fake vs Real News")
plt.xlabel("Class")
plt.ylabel("Number of Articles")
plt.show()

# 8.2 Text length distribution by class
data["text_length"] = data["clean_content"].apply(len)
data.boxplot(column="text_length", by="label")
plt.xticks([1, 2], ["Fake", "Real"])
plt.title("Article Length Distribution by Class")
plt.suptitle("")
plt.xlabel("Class")
plt.ylabel("Text Length")
plt.show()

# ============================================
# 9) Confusion Matrix Plot (Best Model: Logistic Regression)
# ============================================
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_lr,
    display_labels=["Fake", "Real"]
)
plt.title("Confusion Matrix — Logistic Regression")
plt.show()


ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_nb,
    display_labels=["Fake", "Real"]
)
plt.title("Confusion Matrix — Naive Bayes")
plt.show()
