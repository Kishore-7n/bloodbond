
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import joblib  # To save the model

# Load the dataset
data = pd.read_csv("recommendation_data.csv")

# Combine features into a single text column for training
data['combined_features'] = data['Blood Group'] + " " + data['Hemoglobin Level'].astype(str) + " " + data['Disease']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(data['combined_features'], data['Healthy Tips'], test_size=0.2, random_state=42)

# Create a pipeline with TF-IDF vectorizer and Naive Bayes classifier
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save the model for deployment
joblib.dump(model, "health_tips_model.pkl")