import torch
from transformers import BertTokenizer, BertModel
from model.bert_model import BertModelFromScratch

# ------------------------------
# Tokenizer & input
# ------------------------------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

text = """
Artificial intelligence is transforming modern businesses by enabling automation
and improving decision-making across industries.
"""

inputs = tokenizer(
    text,
    return_tensors="pt",
    truncation=True,
    max_length=32
)

# ------------------------------
# HuggingFace BERT (2 layers)
# ------------------------------
hf_model = BertModel.from_pretrained(
    "bert-base-uncased",
    output_hidden_states=True
)
hf_model.encoder.layer = hf_model.encoder.layer[:2]
hf_model.eval()

# ------------------------------
# Scratch BERT (2 layers)
# ------------------------------
scratch_model = BertModelFromScratch(num_layers=2)
scratch_model.eval()

# ------------------------------
# Forward pass
# ------------------------------
with torch.no_grad():
    hf_out = hf_model(**inputs).last_hidden_state
    scratch_out = scratch_model(
        inputs["input_ids"],
        inputs["attention_mask"]
    )

# ------------------------------
# Compare CLS vectors
# ------------------------------
hf_cls = hf_out[:, 0, :]
scratch_cls = scratch_out[:, 0, :]

print("HF CLS shape:", hf_cls.shape)
print("Scratch CLS shape:", scratch_cls.shape)

cos_sim = torch.nn.functional.cosine_similarity(
    hf_cls, scratch_cls
)

print("Cosine similarity:", cos_sim.item())
