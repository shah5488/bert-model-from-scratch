import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel

# ------------------------------
# Simple classifier head
# ------------------------------
class BertClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits


# ------------------------------
# Data
# ------------------------------
texts = [
    """
    Artificial intelligence is transforming modern businesses by enabling automation,
    improving decision making, and optimizing operational efficiency across industries.
    """,

    """
    Sports and physical activities contribute significantly to mental health and
    physical fitness. Team sports promote collaboration and discipline.
    """
]

labels = torch.tensor([0, 1])  # 0 = Tech, 1 = Sports

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

inputs = tokenizer(
    texts,
    padding=True,
    truncation=True,
    max_length=64,
    return_tensors="pt"
)


print(f'inputs tensor keys are \n {inputs.keys()}')

print(f'inputs tensor is \n {inputs}')


# ------------------------------
# Model & forward
# ------------------------------
model = BertClassifier(num_classes=2)
model.eval()

with torch.no_grad():
    logits = model(
        inputs["input_ids"],
        inputs["attention_mask"]
    )

preds = torch.argmax(logits, dim=1)

print("\nLogits:")
print(logits)

print("\nPredicted class:")
print(preds)
