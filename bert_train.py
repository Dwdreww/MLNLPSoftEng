import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score  

import transformers
print("Transformers version in use:", transformers.__version__)
print("Transformers file path:", transformers.__file__)

# -----------------------------
# 1️⃣ Load dataset
# -----------------------------
data = pd.read_csv("train.csv")

# Make sure these columns exist:
# ['comment_text', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

label_cols = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# -----------------------------
# 2️⃣ Dataset class
# -----------------------------
class ToxicDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        # Reset index to avoid KeyError: 0
        self.texts = texts.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])  # safer indexing
        label = torch.tensor(self.labels.iloc[idx].values.astype(float))
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        item = {k: v.squeeze() for k, v in encoding.items()}
        item["labels"] = label
        return item

# -----------------------------
# 3️⃣ Tokenizer and model
# -----------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", 
    num_labels=len(label_cols),
    problem_type="multi_label_classification"
)

# -----------------------------
# 4️⃣ Train-test split
# -----------------------------
train_texts = data['comment_text'][:4000]
test_texts = data['comment_text'][4000:]
train_labels = data[label_cols][:4000]
test_labels = data[label_cols][4000:]

train_dataset = ToxicDataset(train_texts, train_labels, tokenizer)
test_dataset = ToxicDataset(test_texts, test_labels, tokenizer)

# -----------------------------
# 5️⃣ Detect accelerator
# -----------------------------
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
    print("🍎 Using Apple MPS accelerator")
else:
    device = torch.device("cpu")
    print("⚠️ Using CPU (training will be slower).")

model.to(device)

# -----------------------------
# 6️⃣ Training arguments
# -----------------------------
training_args = TrainingArguments(
    output_dir="./bert_toxic_model",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    logging_steps=50
)



# -----------------------------
# 7️⃣ Metrics function
# -----------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = (torch.sigmoid(torch.tensor(logits)) > 0.5).int()
    labels = torch.tensor(labels).int()
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average='micro')
    return {"accuracy": acc, "f1": f1}

# -----------------------------
# 8️⃣ Trainer
# -----------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# -----------------------------
# 9️⃣ Train & Save
# -----------------------------
trainer.train()
model.save_pretrained("./bert_toxic_model_multilabel_final")
tokenizer.save_pretrained("./bert_toxic_model_multilabel_final")

print("✅ Multi-label BERT model trained and saved successfully!")
