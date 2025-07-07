# Clash Royale Card Classifier

# 📦 Step 1: Import Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Replace this with your actual path or upload a CSV to the data folder
df = pd.read_csv(r'C:\Users\finap\OneDrive\Documents\GitHub\FromBIToAI\01_card_classifier\data\clash_royale_cards.csv')

# 👀 Step 3: Explore the Data
print(df.head())
print(df.info())
print(df['rarity'].value_counts())
df['elixirCost'] = df['elixirCost'].fillna(df['elixirCost'].median())


# Encode categorical columns
le = LabelEncoder()
df['rarity_encoded'] = le.fit_transform(df['rarity'])

# Select features and target
features = ['elixirCost', 'Win Rate', 'Usage', 'maxLevel']
X = df[features]
y = df['rarity_encoded']  # Target is common epic rare etc..

# 🧪 Step 5: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🤖 Step 6: Train Model
model = LogisticRegression(max_iter=4000)
model.fit(X_train, y_train)

# 📊 Step 7: Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 🔍 Step 8: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig(r'FromBIToAI\01_card_classifier\outputs')
plt.show()
