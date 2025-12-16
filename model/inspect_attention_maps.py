import torch
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel

# ------------------------------
# Load tokenizer & model
# ------------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained(
    "bert-base-uncased",
    output_attentions=True
)
model.eval()

# ------------------------------
# Input text
# ------------------------------
text = """
Artificial intelligence is transforming modern businesses by enabling automation,
improving decision making, and optimizing customer experiences across industries.
"""

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    max_length=32
)

tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

# ------------------------------
# Forward pass
# ------------------------------
with torch.no_grad():
    outputs = model(**inputs)

attentions = outputs.attentions  # tuple of 12 layers

# ------------------------------
# Select layer & head
# ------------------------------
layer_id = 0        # first transformer layer
head_id = 0         # first attention head

attention_matrix = attentions[layer_id][0, head_id]

# ------------------------------
# CLS attention (row 0)
# ------------------------------
cls_attention = attention_matrix[0]

# ------------------------------
# Plot
# ------------------------------
plt.figure(figsize=(12, 4))
plt.bar(tokens, cls_attention.numpy())
plt.xticks(rotation=90)
plt.title("CLS Token Attention to Other Tokens (Layer 1, Head 1)")
plt.ylabel("Attention weight")
plt.tight_layout()
plt.show()
