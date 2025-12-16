import torch
from transformers import BertTokenizer, BertModel

# ------------------------------
# Load tokenizer and model
# ------------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained(
    "bert-base-uncased",
    output_hidden_states=True,
    output_attentions=True
)

model.eval()

# ------------------------------
# Two text blocks
# ------------------------------
text1 = """
Artificial intelligence is transforming modern businesses by enabling automation,
better decision-making, and improved customer experiences. Companies invest heavily
in data-driven AI systems to gain competitive advantages.
"""

text2 = """
Sports and physical activities contribute significantly to mental health and physical
fitness. Team sports encourage collaboration, while individual sports build discipline
and resilience.
"""

inputs = tokenizer(
    [text1, text2],
    padding=True,
    truncation=True,
    max_length=64,
    return_tensors="pt"
)

# ------------------------------
# Forward pass
# ------------------------------
with torch.no_grad():
    outputs = model(**inputs)

# ------------------------------
# Extract internals
# ------------------------------
hidden_states = outputs.hidden_states   # tuple: embedding + 12 layers
attentions = outputs.attentions         # tuple: 12 layers

# ------------------------------
# Inspect shapes
# ------------------------------
print("\n=== INPUT IDS SHAPE ===")
print(inputs["input_ids"].shape)

print("\n=== EMBEDDING OUTPUT (Layer 0) ===")
print(hidden_states[0].shape)

print("\n=== FIRST TRANSFORMER LAYER OUTPUT (Layer 1) ===")
print(hidden_states[1].shape)

print("\n=== LAST TRANSFORMER LAYER OUTPUT (Layer 12) ===")
print(hidden_states[-1].shape)

# ------------------------------
# CLS token tracking
# ------------------------------
cls_embedding = hidden_states[0][:, 0, :]
cls_after_layer1 = hidden_states[1][:, 0, :]
cls_after_last = hidden_states[-1][:, 0, :]

print("\n=== CLS VECTOR SHAPES ===")
print("CLS at embedding:", cls_embedding.shape)
print("CLS after layer 1:", cls_after_layer1.shape)
print("CLS after last layer:", cls_after_last.shape)

# ------------------------------
# Attention inspection
# ------------------------------
print("\n=== ATTENTION SHAPE (Layer 1) ===")
print(attentions[0].shape)  # (batch, heads, seq_len, seq_len)