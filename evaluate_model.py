import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer
from sklearn.metrics import classification_report
import pandas as pd
from bert_train import ToxicDataset, label_cols  # reuse your dataset class

# -----------------------------
# Load tokenizer & model
# -----------------------------
model_path = "./bert_toxic_model_multilabel_final"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# -----------------------------
# Load your test dataset
# -----------------------------
data = pd.read_csv("train.csv")  # or your real test CSV
test_texts = data['comment_text'][4000:]
test_labels = data[label_cols][4000:]

test_dataset = ToxicDataset(test_texts, test_labels, tokenizer)

# -----------------------------
# Initialize Trainer for evaluation
# -----------------------------
trainer = Trainer(model=model)

# Run evaluation
predictions = trainer.predict(test_dataset)

# -----------------------------
# Convert logits → probabilities → predictions
# -----------------------------
probs = torch.sigmoid(torch.tensor(predictions.predictions))
preds = (probs > 0.5).int().numpy()

# -----------------------------
# Compute metrics
# -----------------------------
print("📊 Classification Report:")
print(classification_report(test_labels, preds, target_names=label_cols, digits=3))
