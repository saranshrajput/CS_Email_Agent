from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from pathlib import Path
import torch, re

ROOT = Path(__file__).resolve().parents[2]
tok = AutoTokenizer.from_pretrained(ROOT/"outputs/bert-email-clf/tokenizer")
mdl = AutoModelForSequenceClassification.from_pretrained(ROOT/"outputs/bert-email-clf/model").eval()

id2label = mdl.config.id2label
def clean(t): return re.sub(r"\s+"," ", t.replace("<br>"," ")).strip()
def predict(t):
    x = tok(clean(t), return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad(): lg = mdl(**x).logits
    p = torch.softmax(lg, -1).squeeze().tolist()
    i = int(torch.argmax(lg, -1))
    labels = [id2label[j] for j in range(len(p))]
    return id2label[i], dict(zip(labels, map(float,p)))
tests = [
  "Subject: Query<br>What are my leads?",
#   "Subject: Request<br>Update my phone number.",
#   "Subject: Help needed<br>Login not working."
]
for t in tests:
    print(predict(t))