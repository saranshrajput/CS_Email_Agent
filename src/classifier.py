import pandas as pd, re
from sklearn.model_selection import train_test_split
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,TrainingArguments, Trainer)
import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd, re
from sklearn.model_selection import train_test_split
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
csv_path = ROOT / "data" / "emails.csv"
df = pd.read_csv(csv_path)

df["email"] = (df["email"]
    .astype(str)
    .str.replace("<br>", " ", regex=False)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

label2id = {"information_query": 0, "actionable": 1, "existing_issue": 2}
id2label = {v:k for k,v in label2id.items()}
df["label_id"] = df["label"].map(label2id)

train_df, tmp_df = train_test_split(df, test_size=0.2, stratify=df["label_id"], random_state=42)
val_df, test_df = train_test_split(tmp_df, test_size=0.5, stratify=tmp_df["label_id"], random_state=42)

model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch["email"], truncation=True, padding="max_length", max_length=256)

train_ds = Dataset.from_pandas(train_df[["email","label_id"]]).rename_column("label_id","labels").map(tokenize, batched=True)
val_ds   = Dataset.from_pandas(val_df[["email","label_id"]]).rename_column("label_id","labels").map(tokenize, batched=True)

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3, id2label=id2label, label2id=label2id)

args = TrainingArguments(
    output_dir="outputs/bert-email-clf",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=4,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="macro"),
    }

trainer = Trainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds, tokenizer=tokenizer, compute_metrics=compute_metrics)
trainer.train()
trainer.save_model("outputs/bert-email-clf/model")
tokenizer.save_pretrained("outputs/bert-email-clf/tokenizer")