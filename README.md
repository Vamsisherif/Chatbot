#  import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch

# Load the dataset
data = pd.read_csv("amazon_reviews.csv")

# Display dataset information
print(data.info())
data.head()
# Check for missing values
print(data.isnull().sum())

# Drop rows with missing values
data.dropna(inplace=True)

# Text Cleaning Function
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # Remove extra whitespace
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text.lower()

# Apply text cleaning
data['review_cleaned'] = data['review_text'].apply(clean_text)

# Encode emotional states
emotions = ['sadness', 'anxiety', 'stress', 'joy']  # Define target labels
data['emotion'] = LabelEncoder().fit_transform(data['emotion'])  # Assuming 'emotion' column exists
data.head()
# Split dataset
X = data['review_cleaned']
y = data['emotion']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to a format suitable for transformers
train_data = pd.DataFrame({'text': X_train, 'label': y_train})
test_data = pd.DataFrame({'text': X_test, 'label': y_test})
train_data.head()
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=4)
def tokenize_data(data):
    return tokenizer(data['text'].tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt")

train_encodings = tokenize_data(train_data)
test_encodings = tokenize_data(test_data)

# Convert labels to tensor
train_labels = torch.tensor(train_data['label'].tolist())
test_labels = torch.tensor(test_data['label'].tolist())
from torch.utils.data import Dataset

class EmotionDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

train_dataset = EmotionDataset(train_encodings, train_labels)
test_dataset = EmotionDataset(test_encodings, test_labels)
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer
)

trainer.train()
def generate_response(emotion_label):
    responses = {
        0: "I'm sorry to hear that. Is there anything I can do to help?",
        1: "Take a deep breath. I'm here to support you.",
        2: "That sounds stressful. How can I assist you?",
        3: "That's wonderful to hear! Tell me more!"
    }
    return responses.get(emotion_label, "I'm here to listen.")
def analyze_sentiment_and_respond(text):
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    output = model(**tokens)
    prediction = torch.argmax(output.logits, dim=1).item()
    response = generate_response(prediction)
    return response

# Test with an example
user_input = "I'm feeling really down today."
response = analyze_sentiment_and_respond(user_input)
print("Chatbot Response:", response)
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break
    response = analyze_sentiment_and_respond(user_input)
    print("Chatbot:", response)
results = trainer.evaluate()
print(results)



